# 看l-AE，l-VAE学习到的latent space是否具有距离可分性
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from data_load import *  # Step 1: 读取和预处理数据
import torch.optim as optim
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
import time

"""
'cicids_custom': ['Tuesday'], 30
'toniot_custom': ['ddos'], 30
'cse_improved': ['server1'], 40
'Graph': ['ethereum'], 34
'BNaT': ['w1']  19
"""

# Step 1: 读取和预处理数据
dataset = 'toniot_custom'
subset = 'ddos'
X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

# 合并训练集、验证集和测试集
X_all = np.concatenate([X_train, X_valid, X_test], axis=0)
Y_all = np.concatenate([y_train, y_valid, y_test], axis=0)

# 创建对象，按列归一化数据
scaler = StandardScaler()
# 对合并后的数据进行归一化
X_all = scaler.fit_transform(X_all)

# 划分正常/异常样本
X_normal = X_all[Y_all == 0]

# 转为 tensor
X_normal_tensor = torch.tensor(X_normal, dtype=torch.float32)
X_all_tensor = torch.tensor(X_all, dtype=torch.float32)


# Step 2: 定义 l - AE 模型
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded  # 返回解码结果和潜在空间表示


# Step 3: 定义潜在空间距离损失
def latent_space_distance_loss(latent_representation):
    """计算潜在空间中样本与均值的距离损失"""
    mean_latent = latent_representation.mean(dim=0)  # 计算潜在空间的均值
    distance = torch.norm(latent_representation - mean_latent, dim=1)  # 计算每个样本的距离
    return distance.mean()  # 返回所有样本与均值的平均距离作为损失


# Step 4: 训练 l - AE
input_dim = X_train.shape[1]
l_ae = Autoencoder(input_dim)
criterion = nn.MSELoss()  # 重构误差损失
optimizer_l_ae = optim.Adam(l_ae.parameters(), lr=0.001)

num_epochs = 50
latent_loss_weight = 0.1

start_train_time = time.time()  # 记录训练开始时间

for epoch in range(num_epochs):
    l_ae.train()
    train_loss = 0
    latent_loss = 0
    outputs, latent_representation = l_ae(X_normal_tensor)

    # 计算重构误差
    reconstruction_loss = criterion(outputs, X_normal_tensor)

    # 计算潜在空间的距离损失
    distance_loss = latent_space_distance_loss(latent_representation)

    # 总损失 = 重构误差 + 潜在空间距离损失
    total_loss = reconstruction_loss + distance_loss
    optimizer_l_ae.zero_grad()
    total_loss.backward()
    optimizer_l_ae.step()

    train_loss += reconstruction_loss.item()
    latent_loss += distance_loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Reconstruction Loss: {train_loss:.4f}, Latent Loss: {latent_loss:.4f}")

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时长


# Step 5: 定义 l - VAE 模型
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # 编码器部分：输入x得到均值(mu)和对数标准差(logvar)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # 均值和对数方差
        self.fc_mu = nn.Linear(8, latent_dim)  # 均值
        self.fc_logvar = nn.Linear(8, latent_dim)  # 对数方差

        # 解码器部分：从潜在空间z生成输出
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()  # 输出值在[0, 1]之间
        )

    def reparameterize(self, mu, logvar):
        """通过重参数化技巧从分布中采样"""
        std = torch.exp(0.5 * logvar)  # 标准差
        epsilon = torch.randn_like(std)  # 从标准正态分布中采样
        z = mu + epsilon * std  # 重参数化公式
        return z

    def forward(self, x):
        """VAE的前向传播"""
        h1 = self.encoder(x)
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)

        # 使用重参数化技巧生成潜在空间的z
        z = self.reparameterize(mu, logvar)

        # 解码器生成重构数据
        reconstructed_x = self.decoder(z)

        return reconstructed_x, mu, logvar


# 定义VAE的损失函数（重构误差 + KL散度）
import torch.nn.functional as F


def vae_loss_function(reconstructed_x, x, mu, logvar):
    # 重构损失：均方误差
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL 散度损失：衡量潜在变量的分布与标准正态分布的差异
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_loss


# Step 6: 训练 l - VAE
latent_dim = 8  # 设置潜在空间维度
l_vae = VAE(input_dim, latent_dim)
optimizer_l_vae = optim.Adam(l_vae.parameters(), lr=0.001)

start_train_time = time.time()  # 记录训练开始时间
# 训练
num_epochs = 50

for epoch in range(num_epochs):
    l_vae.train()
    train_loss = 0
    latent_loss = 0
    outputs, mu, logvar = l_vae(X_normal_tensor)

    # 计算重构误差
    reconstruction_loss = vae_loss_function(outputs, X_normal_tensor, mu, logvar)

    # 计算潜在空间的距离损失
    distance_loss = latent_space_distance_loss(mu)  # 使用潜在空间的均值来计算距离损失

    # 总损失 = 重构误差 + 潜在空间距离损失
    total_loss = reconstruction_loss + distance_loss
    optimizer_l_vae.zero_grad()
    total_loss.backward()
    optimizer_l_vae.step()

    train_loss += reconstruction_loss.item()
    latent_loss += distance_loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Reconstruction Loss: {train_loss:.4f}, Latent Loss: {latent_loss:.4f}")

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时长
print(f"VAE - improve Training Time: {train_time:.6f} seconds")

# Step 7: 推理并可视化 latent space
l_ae.eval()
_, z_l_ae = l_ae(X_all_tensor)
z_l_ae = z_l_ae.detach().numpy()

l_vae.eval()
_, mu, _ = l_vae(X_all_tensor)
z_l_vae = mu.detach().numpy()  # 取 mu 表示 latent center

# 计算 l - AE 和 l - VAE latent 空间中每个点到正常样本中心的距离
center_l_ae = z_l_ae[Y_all == 0].mean(axis=0)
dists_l_ae = np.linalg.norm(z_l_ae - center_l_ae, axis=1)

center_l_vae = z_l_vae[Y_all == 0].mean(axis=0)
dists_l_vae = np.linalg.norm(z_l_vae - center_l_vae, axis=1)

# 可视化 l - AE latent distance 分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(dists_l_ae[Y_all == 0], label='Normal', fill=True, color='blue')
sns.kdeplot(dists_l_ae[Y_all == 1], label='Anomaly', fill=True, color='red')
plt.title("SLS(AE): Latent Distance Distribution")
plt.xlabel("Distance to center")
plt.xlim(0, 0.8)
plt.legend()

# 可视化 l - VAE latent distance 分布
plt.subplot(1, 2, 2)
sns.kdeplot(dists_l_vae[Y_all == 0], label='Normal', fill=True, color='blue')
sns.kdeplot(dists_l_vae[Y_all == 1], label='Anomaly', fill=True, color='red')
plt.title("SLS(VAE): Latent Distance Distribution")
plt.xlabel("Distance to center")
plt.xlim(0, 0.8)
plt.legend()

plt.tight_layout()
plt.show()
# img_name = f"{1}.png"
# plt.savefig(img_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
