import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from scipy.stats import chi2
import time
from data_load import *
from utils import *

# load data
"""
'cicids_custom': ['Tuesday'],
'toniot_custom': ['ddos'], 
'cse_improved': ['server1'], 
'Graph': ['ethereum'], 
'BNaT': ['w1']  
"""

dataset = 'cicids_custom'
subset = 'Tuesday'

X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

# 创建MinMaxScaler对象，按列归一化数据
scaler = MinMaxScaler()

# 对训练集、验证集和测试集的特征进行归一化
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)  # 使用训练集的scaler进行变换
X_test = scaler.transform(X_test)  # 使用训练集的scaler进行变换


# 转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # 均值 & 对数方差
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        h1 = self.encoder(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)

        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, logvar

def vae_loss_function(reconstructed_x, x, mu, logvar):
    # 重构损失: MSE
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')
    # KL散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_loss

# ------------------------------
# Step 4: 改进的潜在空间损失 (可选：带方差惩罚)
# ------------------------------
def latent_space_distance_loss(mu_batch, alpha=0.1):
    """
    (1) 对 batch 内所有样本均值的距离惩罚
    (2) 对 batch 内方差进行惩罚 (alpha * var)
    """
    mean_latent = mu_batch.mean(dim=0)
    dist = torch.norm(mu_batch - mean_latent, dim=1).mean()

    # 计算简单的方差(所有维度的平均)
    var = ((mu_batch - mean_latent)**2).mean()

    # total
    loss_val = dist + alpha * var
    return loss_val

# ------------------------------
# Step 5: 训练模型
# ------------------------------
input_dim = X_train.shape[1]
latent_dim = 8
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
alpha = 0.1  # 方差惩罚系数

start_train_time = time.time()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    latent_loss_val = 0.0

    for batch in train_loader:
        inputs = batch[0]
        outputs, mu, logvar = model(inputs)

        # 基础 VAE 损失
        reconstruction_loss = vae_loss_function(outputs, inputs, mu, logvar)
        # 潜在聚合损失
        dist_loss = latent_space_distance_loss(mu, alpha=alpha)

        total_loss = reconstruction_loss + dist_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += reconstruction_loss.item()
        latent_loss_val += dist_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Recon Loss: {train_loss/len(train_loader):.4f}, "
          f"Latent Loss: {latent_loss_val/len(train_loader):.4f}")
end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时长
print(f"VAE-improve Training Time: {train_time:.6f} seconds")
# ------------------------------
# Step 6: 获取潜在表示
# ------------------------------
@torch.no_grad()
def get_latent_space_representation(model, data_tensor):
    """
    只返回潜在均值 mu
    """
    _, mu, _ = model(data_tensor)
    return mu

X_train_latent = get_latent_space_representation(model, X_train_tensor)
X_valid_latent = get_latent_space_representation(model, X_valid_tensor)
X_test_latent  = get_latent_space_representation(model, X_test_tensor)

# ------------------------------
# Step 7: 计算全局均值和协方差, 进行马氏距离判别
# ------------------------------
normal_train_latent = X_train_latent  # 训练集中的正常样本

mu_overall = normal_train_latent.mean(dim=0)

# 若 PyTorch 版本支持 torch.cov (>=1.10):
diff = normal_train_latent - mu_overall
Sigma = torch.cov(diff.T)  # (D x D)

# 若协方差矩阵奇异，可加一点噪声
eps = 1e-6
Sigma = Sigma + eps * torch.eye(latent_dim)
Sigma_inv = torch.inverse(Sigma)

# 如果 torch.cov 不可用, 可转 numpy:
# diff_np = diff.cpu().numpy()
# Sigma_np = np.cov(diff_np.T)
# Sigma = torch.from_numpy(Sigma_np).float() + eps * torch.eye(latent_dim)
# Sigma_inv = torch.inverse(Sigma)

def mahalanobis_distance(z, mu, inv_cov):
    diff_z = (z - mu).unsqueeze(1)  # (D) -> (D,1)
    md = diff_z.T @ inv_cov @ diff_z
    return md.squeeze()

# 这里用卡方分布的临界值模拟 "3-sigma"
threshold = chi2.ppf(0.997, df=latent_dim)  # 约 99.7% 的覆盖率

from sklearn.metrics import classification_report

# ------------------------------
# Step 8: 在验证集上进行异常检测并输出分类报告
# ------------------------------
valid_md = []
for z in X_valid_latent:
    md_val = mahalanobis_distance(z, mu_overall, Sigma_inv)
    valid_md.append(md_val.item())

valid_md = np.array(valid_md)
valid_preds = (valid_md > threshold).astype(float)  # 1=异常, 0=正常

# # 修改处：直接输出分类报告
# print("\n[Validation Set Classification Report]")
# print(classification_report(y_valid, valid_preds, target_names=["Normal", "Anomaly"]))


# ------------------------------
# Step 9: 在测试集上进行异常检测并输出分类报告
# ------------------------------
start_test_time = time.time()  # 记录训练开始时间
test_md = []
for z in X_test_latent:
    md_val = mahalanobis_distance(z, mu_overall, Sigma_inv)
    test_md.append(md_val.item())

test_md = np.array(test_md)
test_preds = (test_md > threshold).astype(float)
end_test_time = time.time()
test_time = end_test_time - start_test_time
print(f"VAE-improve Testing Time: {test_time:.6f} seconds")

# # 修改处：直接输出分类报告
# print("\n[Test Set Classification Report]")
# print(classification_report(y_test, test_preds, target_names=["Normal", "Anomaly"]))

evaluate_predictions(y_test, test_preds)
