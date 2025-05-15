import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from src.data_load import load_data
import time
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


SEED = 42
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 50
BATCH_SIZE = 64
LR = 0.001
FPR_list = np.arange(0.001, 0.051, 0.001)
NORMALIZER_DIR = 'normalizers'
TARGET_MODEL_DIR = 'models'


def mse_each(x, x_rec):
    return torch.square(x - x_rec).mean(dim=1)


def TP(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum().item()


def FP(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 1)).sum().item()


def TN(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum().item()


def FN(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 0)).sum().item()


class ActiveAutoEncoder(nn.Module):
    def __init__(self, n_feat):
        super(ActiveAutoEncoder, self).__init__()
        self.n_feat = n_feat
        self.encoder = nn.Sequential(
            nn.Linear(self.n_feat, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.n_feat),
            nn.LeakyReLU(0.2),
        )
        self.thres = 0

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss_func(self, x, recons):
        return F.mse_loss(x, recons)

    def loss_func_each(self, x, x_rec):
        return torch.square(x - x_rec).mean(dim=1).detach()

    def score_samples(self, X, cuda=True):
        if cuda:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).to(DEVICE).float()
            elif isinstance(X, torch.Tensor):
                X = X.to(DEVICE).float()
            else:
                raise ValueError("Unsupported input type")
        else:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif isinstance(X, torch.Tensor):
                X = X.float()
            else:
                raise ValueError("Unsupported input type")
        _, recons = self.forward(X)
        return self.loss_func_each(X, recons).cpu().numpy().reshape(-1, )

    def predict(self, X):
        return (self.score_samples(X) > self.thres).astype(int)

    def active_learning_influence(self, X, y):
        # 基于影响力的主动学习示例
        # 计算每个样本的梯度范数作为影响力
        X.requires_grad = True
        _, recons = self.forward(X)
        loss = self.loss_func(X, recons)
        self.zero_grad()
        loss.backward()
        grad_norms = torch.norm(X.grad, dim=1)
        # 根据梯度范数选择影响力大的样本
        top_k = int(0.1 * len(X))  # 选择前10%的样本
        top_indices = torch.topk(grad_norms, top_k)[1]
        X_selected = X[top_indices]
        y_selected = y[top_indices] if y is not None else None
        return X_selected, y_selected

    def expansion_shrinkage_operator(self, X, y):
        # 扩展收缩算子示例
        # 根据样本的重建误差调整样本权重
        _, recons = self.forward(X)
        errors = self.loss_func_each(X, recons)
        # 定义扩展收缩的阈值
        threshold = errors.mean()
        weights = torch.where(errors > threshold, 1.2, 0.8)
        X_weighted = X * weights.unsqueeze(1)
        return X_weighted, y


def train_process(dataset, subset):
    start_time = time.time()
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    n_feat = X_train.shape[1]
    print(X_train.shape)
    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train = np.array(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    if not os.path.exists(NORMALIZER_DIR):
        os.makedirs(NORMALIZER_DIR)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}_columns.pkl'), 'wb') as f:
        pickle.dump(X_train.columns, f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
        pickle.dump(normalizer, f)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)
    aae = ActiveAutoEncoder(n_feat=n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(aae.parameters(), lr=LR)
    loss_func = nn.MSELoss().to(DEVICE)
    for epoch in range(EPOCH):
        for i, (x,) in enumerate(train_loader):
            x = x.to(DEVICE)
            # 应用扩展收缩算子
            x_weighted, _ = aae.expansion_shrinkage_operator(x, None)
            _, x_rec = aae(x_weighted)
            loss_train = loss_func(x_weighted, x_rec)
            # 应用基于影响力的主动学习
            x_selected, _ = aae.active_learning_influence(x_weighted, None)
            _, x_rec_selected = aae(x_selected)
            loss_selected = loss_func(x_selected, x_rec_selected)
            total_loss = loss_train + loss_selected
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch :', epoch, '|', f'train_loss:{total_loss.data}')

    aae.eval()
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.to(DEVICE)
            _, x_rec = aae(x)
            mse = mse_each(x, x_rec)
            y[y != 0] = 1
            mse_list.append(mse)
            y_list.append(y)
    mse_all = torch.cat(mse_list).view(-1, )
    y_true = torch.cat(y_list).view(-1, )
    mse_neg = mse_all[y_true == 0]

    best_thres, best_score = None, 0
    for FPR in FPR_list:
        thres = mse_neg.quantile(1 - FPR).item()
        y_pred = (mse_all > thres).view(-1, ).int().cpu()
        tp = TP(y_true, y_pred)
        fp = FP(y_true, y_pred)
        fn = FN(y_true, y_pred)
        recall = tp / (tp + fn)
        prec = tp / (tp + fp)
        score = 2 * recall * prec / (recall + prec)
        if score > best_score:
            print('FPR', FPR, 'score', score)
            best_thres = thres
            best_score = score
            best_percentile = 1 - FPR  # 保存对应的百分位数
    aae.thres = best_thres

    print(f"Best threshold: {best_thres}, Corresponding percentile: {best_percentile}")  # 打印最优阈值和对应的百分位数

    if not os.path.exists(TARGET_MODEL_DIR):
        os.makedirs(TARGET_MODEL_DIR)
    torch.save(aae, os.path.join(TARGET_MODEL_DIR, f'AAE_{dataset}_{subset}.model'))
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")


def test_process(dataset, subset):
    start_time = time.time()
    aae = torch.load(os.path.join(TARGET_MODEL_DIR, f'AAE_{dataset}_{subset}.model')).to(DEVICE)
    aae.eval()
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    thres = aae.thres
    X, y = load_data(dataset, subset, mode='test')
    # 确保 X 和 y 是 numpy.ndarray 类型
    X = np.array(X)
    y = np.array(y)
    # 检查数据是否包含 NaN
    if np.isnan(X).any() or np.isnan(y).any():
        print("Data contains NaN values.")
        X = np.nan_to_num(X, nan=0)
        y = np.nan_to_num(y, nan=0)
    print(f"Test data shape before normalization: {X.shape}")
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}_columns.pkl'), 'rb') as f:
        train_columns = pickle.load(f)
    X = pd.DataFrame(X, columns=train_columns)  # 为测试数据赋予训练数据的列顺序
    X = X.reindex(columns=train_columns, fill_value=0)  # 填充缺失列
    print(f"Test data shape after reindexing: {X.shape}")
    X = normalizer.transform(X)
    print(f"Test data shape after normalization: {X.shape}")
    test_set = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, drop_last=True)
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            _, x_rec = aae(x)
            mse = mse_each(x, x_rec).cpu()
            # 检查 MSE 是否包含 NaN
            if torch.isnan(mse).any():
                print("MSE contains NaN values.")
                mse = torch.nan_to_num(mse, nan=0)
            y[y != 0] = 1
            mse_list.extend(mse.numpy())
            y_list.extend(y.numpy())
    y_pred = (np.array(mse_list) > thres).astype(int)
    tp = TP(np.array(y_list), y_pred)
    fp = FP(np.array(y_list), y_pred)
    tn = TN(np.array(y_list), y_pred)
    fn = FN(np.array(y_list), y_pred)
    tpr = recall_score(y_list, y_pred)
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = precision_score(y_list, y_pred)
    f1 = f1_score(y_list, y_pred)
    auroc = roc_auc_score(y_list, mse_list)
    aupr = average_precision_score(y_list, mse_list)
    # 计算 FPR95
    precision_list, recall, _ = precision_recall_curve(y_list, mse_list)
    fpr95 = None
    for i in range(len(recall)):
        if recall[i] >= 0.95:
            fpr95 = fp / (fp + tn)
            break
    end_time = time.time()
    testing_time = end_time - start_time
    print(f"Testing time: {testing_time} seconds")
    print('TPR:', tpr)
    print('TNR:', tnr)
    print('Precision:', precision)
    print('F1:', f1)
    print('AUROC:', auroc)
    print('AUPR:', aupr)
    print('FPR95:', fpr95)

"""
'cicids_custom': ['Tuesday'],
'toniot_custom': ['ddos'], 
'cse_improved': ['server1'], 
'Graph': ['ethereum'], 
'BNaT': ['w1']  
"""

train_process('Graph', 'ethereum')
test_process('Graph', 'ethereum')
