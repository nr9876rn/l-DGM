import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import time
from sklearn.metrics import precision_recall_curve
from normalize import *
from utils import *
from src.data_load import load_data

SEED = 42
torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 50
BATCH_SIZE = 64
LR = 0.001
FPR_list = np.arange(0.001, 0.791, 0.001)


class RDAutoEncoder(nn.Module):
    def __init__(self, n_feat):
        super(RDAutoEncoder, self).__init__()
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
        self.outlier_decoder = nn.Sequential(
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
        outlier = self.outlier_decoder(encoded)
        clean = x - outlier
        return encoded, decoded, clean, outlier

    def loss_func(self, x, recons, outlier):
        clean = x - outlier
        clean_loss = F.mse_loss(clean, recons)
        outlier_loss = F.mse_loss(outlier, torch.zeros_like(outlier))
        return clean_loss + outlier_loss

    def loss_func_each(self, x, x_rec, outlier):
        clean = x - outlier
        clean_loss = torch.square(clean - x_rec).mean(dim=1)
        outlier_loss = torch.square(outlier).mean(dim=1)
        return (clean_loss + outlier_loss).detach()

    def score_samples(self, X, cuda=True):
        if cuda:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            X = torch.from_numpy(X).to(DEVICE).float()
        else:
            X = torch.from_numpy(X).float()
        _, _, _, outlier = self.forward(X)
        return torch.square(outlier).mean(dim=1).cpu().numpy().reshape(-1, )

    def predict(self, X):
        return (self.score_samples(X) > self.thres).astype(int)


def train_process(dataset, subset):
    start_time = time.time()
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    n_feat = X_train.shape[1]
    print(X_train.shape)
    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train = pd.DataFrame(X_train)
    X_train.columns = [f"feature_{i}" for i in range(X_train.shape[1])]
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}_columns.pkl'), 'wb') as f:
        pickle.dump(X_train.columns, f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
        pickle.dump(normalizer, f)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
    train_set = TensorDataset(torch.from_numpy(X_train).float())
    eval_set = TensorDataset(torch.from_numpy(X_eval).float(), torch.from_numpy(y_eval).float())
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, drop_last=True)
    rdae = RDAutoEncoder(n_feat=n_feat).to(DEVICE)
    optimizer = torch.optim.Adam(rdae.parameters(), lr=LR)
    for epoch in range(EPOCH):
        for i, (x,) in enumerate(train_loader):
            x = x.to(DEVICE)
            _, x_rec, clean, outlier = rdae(x)
            loss_train = rdae.loss_func(x, x_rec, outlier)
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if i % 100 == 0:
                print('Epoch :', epoch, '|', f'train_loss:{loss_train.data}')

    rdae.eval()
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(eval_loader):
            x = x.to(DEVICE)
            _, _, _, outlier = rdae(x)
            mse = torch.square(outlier).mean(dim=1).cpu()
            y[y != 0] = 1
            mse_list.append(mse)
            y_list.append(y)
    mse_all = torch.concat(mse_list).view(-1, )
    y_true = torch.concat(y_list).view(-1, )
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
    rdae.thres = best_thres

    print(f"Best threshold: {best_thres}, Corresponding percentile: {best_percentile}")  # 打印最优阈值和对应的百分位数

    torch.save(rdae, os.path.join(TARGET_MODEL_DIR, f'RDAE_{dataset}_{subset}.model'))
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training time: {training_time} seconds")
    return training_time


def test_process(dataset, subset):
    start_time = time.time()
    rdae = torch.load(os.path.join(TARGET_MODEL_DIR, f'RDAE_{dataset}_{subset}.model')).to(DEVICE)
    rdae.eval()
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)
    thres = rdae.thres
    X, y = load_data(dataset, subset, mode='test')
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

    tp, fp, tn, fn = 0, 0, 0, 0
    mse_list, y_list = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x = x.to(DEVICE)
            _, _, _, outlier = rdae(x)
            mse = torch.square(outlier).mean(dim=1).cpu()
            y_pred = (mse > thres).view(-1, ).int().cpu()
            y[y != 0] = 1
            tp += TP(y, y_pred)
            fp += FP(y, y_pred)
            tn += TN(y, y_pred)
            fn += FN(y, y_pred)
            mse_list.extend(mse.numpy())
            y_list.extend(y.numpy())
    # 处理 NaN 值
    mse_list = np.array(mse_list)
    y_list = np.array(y_list)
    valid_indices = ~np.isnan(mse_list)
    mse_list = mse_list[valid_indices]
    y_list = y_list[valid_indices]

    if len(mse_list) == 0 or len(y_list) == 0:
        print("Warning: All samples were filtered out due to NaN values.")
        tpr, tnr, precision, f1, auc, aupr, fpr95 = 0, 0, 0, 0, 0, 0, 0
    else:
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0
        auc = roc_auc_score(y_list, mse_list)
        aupr = average_precision_score(y_list, mse_list)

        precision_curve, recall_curve, _ = precision_recall_curve(y_list, mse_list)
        fpr95 = None
        for i in range(len(recall_curve)):
            if recall_curve[i] >= 0.95:
                fpr95 = 1 - tnr
                break

    end_time = time.time()
    testing_time = end_time - start_time

    print('Training time:', train_process(dataset, subset))
    print('Testing time:', testing_time)
    print('TPR:', tpr)
    print('TNR:', tnr)
    print('Precision:', precision)
    print('F1:', f1)
    print('AUROC:', auc)
    print('AUPR:', aupr)
    print('FPR95:', fpr95)
    save_result({
        'dataset': dataset, 'subset': subset,
        'TPR': tpr, 'TNR': tnr, 'Precision': precision, 'F1': f1,
        'AUROC': auc, 'AUPR': aupr, 'FPR95': fpr95
    }, 'RDAE')


"""
'cicids_custom': ['Tuesday'],
'toniot_custom': ['ddos'], 
'cse_improved': ['server1'], 
'Graph': ['ethereum'], 
'BNaT': ['w1']  
"""

# 训练+测试
train_process('cicids_custom', 'Tuesday')
test_process('cicids_custom', 'Tuesday')
