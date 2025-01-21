# 卿雨竹
# 开发时间：2024-12-30 1:26
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from data_load import *
from utils import *

torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
"""
'cicids_custom': ['Tuesday'],
'toniot_custom': ['ddos'], 
'cse_improved': ['server1'], 
'Graph': ['ethereum'], 
'BNaT': ['w1']  
"""

dataset = 'BNaT'
subset = 'w1'
X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')
with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
    normalizer = pickle.load(f)
X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
X_test = normalizer.transform(X_test)

# 将X_train, X_eval转换为Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_eval_tensor = torch.tensor(X_eval, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)


# 将数据加载为DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
eval_dataset = TensorDataset(X_eval_tensor, y_eval_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=32)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # 输出重建的输入
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded  # 返回重建结果和编码特征

class PFA(nn.Module):
    def __init__(self, input_dim):
        super(PFA, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, feature):
        # 假设是一个简单的线性变换，可以根据需要调整
        adapted_feature = self.fc(feature)
        return adapted_feature

class CAS(nn.Module):
    def __init__(self, input_dim):
        super(CAS, self).__init__()
        self.fc = nn.Linear(input_dim, 1)  # 生成像素级的异常得分

    def forward(self, feature):
        score = self.fc(feature)
        return score


class PCSNetWithAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PCSNetWithAE, self).__init__()
        self.autoencoder = Autoencoder(input_dim, hidden_dim)
        self.pfa = PFA(hidden_dim // 4)  # 假设特征维度是 encoder 的输出维度
        self.cas = CAS(hidden_dim // 4)

    def forward(self, x):
        # AE 前向传播
        decoded, encoded = self.autoencoder(x)

        # PFA 适配特征
        adapted_feature = self.pfa(encoded)

        # CAS 进行像素级异常定位
        anomaly_score = self.cas(adapted_feature)

        return decoded, anomaly_score  # 输出重建结果和异常得分

def train_model(model, train_loader, epochs):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)

            optimizer.zero_grad()

            # 前向传播
            decoded, anomaly_score = model(X_batch)
            reconstruction_loss = mse_loss(decoded, X_batch)  # AE 重建误差
            anomaly_loss = ce_loss(anomaly_score, y_batch)  # 假设 y_batch 是正常/异常标签

            loss = reconstruction_loss + anomaly_loss
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}")


def detect_anomalies(model, X_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test).float().to(DEVICE)
        decoded, anomaly_score = model(X_test_tensor)
        # 选择 95% 分位数作为阈值
        threshold = np.percentile(anomaly_score, 99)
        print(f"Chosen threshold (99th percentile): {threshold}")

        print(f"test Anomaly scores: {anomaly_score[:10]}")  # 打印前 10 个异常分数
        print(f"test Anomaly scores range: {anomaly_score.min()} to {anomaly_score.max()}")
        # 计算重建误差
        reconstruction_error = torch.mean((X_test_tensor - decoded) ** 2, dim=1)

        # 将得分转换为异常标记
        anomalies = reconstruction_error > threshold
        return anomalies.cpu().numpy(), reconstruction_error.cpu().numpy()

start_train_time_blackbox = time.time()  # 记录测试开始时间
input_dim = X_train.shape[1]  # 输入特征的维度
hidden_dim = 128  # 可根据需要调整隐藏层维度
model = PCSNetWithAE(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)

train_model(model, train_loader, epochs=50)  # 调用训练函数
train_time_blackbox = time.time() - start_train_time_blackbox
print(f"Blackbox Model Training Time: {train_time_blackbox:.4f} seconds")


def evaluate_model(model, X_test, y_test):
    anomalies, _ = detect_anomalies(model, X_test)
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, anomalies).ravel()
    # 计算 Precision, Recall (TPR), Specificity (TNR), F1 score, FPR, FNR
    precision = precision_score(y_test, anomalies)
    recall = recall_score(y_test, anomalies)  # TPR
    specificity = tn / (tn + fp)  # TNR
    f1 = f1_score(y_test, anomalies)
    fpr = fp / (tn + fp)  # FPR
    fnr = fn / (fn + tp)  # FNR

    # 打印结果
    print(f"Precision: {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print(f"Anomalies detected: {sum(anomalies)} out of {len(anomalies)}")
    return precision, recall, specificity, f1, fpr, fnr


start_test_time_blackbox = time.time()  # 记录测试开始时间
evaluate_model(model, X_test, y_test)
test_time_blackbox = time.time() - start_test_time_blackbox  # 计算测试时间
print(f"Blackbox Model Testing Time: {test_time_blackbox:.4f} seconds")