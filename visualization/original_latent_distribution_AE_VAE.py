# 看AE，VAE学习到的latent space是否具有距离可分性
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from data_load import *

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

# Step 2: 定义 AE / VAE 模型
class AE(nn.Module):
    def __init__(self, input_dim=30, latent_dim=8):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z


class VAE(nn.Module):
    def __init__(self, input_dim=30, latent_dim=8):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc21 = nn.Linear(16, latent_dim)  # mu
        self.fc22 = nn.Linear(16, latent_dim)  # logvar
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


# Step 3: 训练 AE
ae = AE()
optimizer_ae = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(20):
    ae.train()
    out, _ = ae(X_normal_tensor)
    loss = loss_fn(out, X_normal_tensor)
    optimizer_ae.zero_grad()
    loss.backward()
    optimizer_ae.step()

# Step 4: 训练 VAE
vae = VAE()
optimizer_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)


def vae_loss_fn(x, x_recon, mu, logvar):
    recon_loss = nn.MSELoss()(x_recon, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + kl


for epoch in range(20):
    vae.train()
    x_recon, mu, logvar, _ = vae(X_normal_tensor)
    loss = vae_loss_fn(X_normal_tensor, x_recon, mu, logvar)
    optimizer_vae.zero_grad()
    loss.backward()
    optimizer_vae.step()

# Step 5: 推理并可视化 latent space
ae.eval()
_, z_ae = ae(X_all_tensor)
z_ae = z_ae.detach().numpy()

vae.eval()
_, mu, _, _ = vae(X_all_tensor)
z_vae = mu.detach().numpy()  # 取 mu 表示 latent center

# 计算 AE 和 VAE latent 空间中每个点到正常样本中心的距离
center_ae = z_ae[Y_all == 0].mean(axis=0)
dists_ae = np.linalg.norm(z_ae - center_ae, axis=1)

center_vae = z_vae[Y_all == 0].mean(axis=0)
dists_vae = np.linalg.norm(z_vae - center_vae, axis=1)

# 可视化 AE latent distance 分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.kdeplot(dists_ae[Y_all == 0], label='Normal', fill=True, color='blue')
sns.kdeplot(dists_ae[Y_all == 1], label='Anomaly', fill=True, color='red')
plt.title("AE: Latent Distance Distribution")
plt.xlabel("Distance to center")
plt.xlim(0, 5)
plt.legend()

# 可视化 VAE latent distance 分布
plt.subplot(1, 2, 2)
sns.kdeplot(dists_vae[Y_all == 0], label='Normal', fill=True, color='blue')
sns.kdeplot(dists_vae[Y_all == 1], label='Anomaly', fill=True, color='red')
plt.title("VAE: Latent Distance Distribution")
plt.xlabel("Distance to center")
plt.xlim(0, 5)
plt.legend()

plt.tight_layout()
plt.show()
# img_name = f"{1}.png"
# plt.savefig(img_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
