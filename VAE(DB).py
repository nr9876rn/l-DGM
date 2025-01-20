# 模型训练没加 latent loss的VAE-improve
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import torch.nn.functional as F
from eval_output import output_report
import time

# Step 1: 数据读取
file_path = r'C:\Users\nr\Desktop\graduation project\UAD-Rule-Extraction-main\dataset\Graph\ethereum.txt'
data = pd.read_csv(file_path, header=None)

# 提取用户ID、标签和数据值
user_ids = data[0]
labels = data[35]
features = data.iloc[:, 1:35]

# Step 2: 保持标签比例的100%采样
data_sampled = data.groupby(35, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(
    drop=True)

# 更新采样后的特征和标签
sampled_features = data_sampled.iloc[:, 1:35]
sampled_labels = data_sampled[35]

# Step 3: 对1-34列进行归一化
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(sampled_features)

# Step 4: 数据集划分
# 将数据分为训练集 (train + valid) 和测试集 (test)
normal_data = normalized_features[sampled_labels == 0]
anomaly_data = normalized_features[sampled_labels == 1]

# 划分训练集和测试集
normal_train_valid, normal_test = train_test_split(normal_data, test_size=0.2, random_state=42)
anomaly_train_valid, anomaly_test = train_test_split(anomaly_data, test_size=0.2, random_state=42)

# 从训练集中划分验证集
normal_train, normal_valid = train_test_split(normal_train_valid, test_size=0.2, random_state=42)
anomaly_train, anomaly_valid = train_test_split(anomaly_train_valid, test_size=0.2, random_state=42)

# 训练集：仅正常样本
X_train = normal_train

# 验证集：正常样本 + 异常样本
X_valid = np.vstack((normal_valid, anomaly_valid))
y_valid = np.hstack((np.zeros(len(normal_valid)), np.ones(len(anomaly_valid))))

# 测试集：正常样本 + 异常样本
X_test = np.vstack((normal_test, anomaly_test))
y_test = np.hstack((np.zeros(len(normal_test)), np.ones(len(anomaly_test))))

# 转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)


# 定义VAE模型
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
def vae_loss_function(reconstructed_x, x, mu, logvar):
    # 重构损失：均方误差
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

    # KL 散度损失：衡量潜在变量的分布与标准正态分布的差异
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + kl_loss


# 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
latent_dim = 8  # 设置潜在空间维度
model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 7: 训练模型
num_epochs = 50
latent_loss_weight = 0.1

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    latent_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        outputs, mu, logvar = model(inputs)

        # 计算重构误差
        reconstruction_loss = vae_loss_function(outputs, inputs, mu, logvar)

        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()

        train_loss += reconstruction_loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Reconstruction Loss: {train_loss / len(train_loader):.4f}, Latent Loss: {latent_loss / len(train_loader):.4f}")


# Step 8: 获取潜在空间表示
def get_latent_space_representation(model, data_tensor):
    with torch.no_grad():
        _, mu, _ = model(data_tensor)  # 仅获取潜在空间的均值表示
    return mu


# 获取训练集和验证集的潜在空间表示
X_train_latent = get_latent_space_representation(model, X_train_tensor)
X_valid_latent = get_latent_space_representation(model, X_valid_tensor)
X_test_latent = get_latent_space_representation(model, X_test_tensor)

# Step 9: 计算正常样本的特征中心 (μ)
normal_train_latent = X_train_latent  # 仅使用标签为0的正常样本
mu_overall = normal_train_latent.mean(dim=0)  # 特征中心 μ
std_overall = normal_train_latent.std(dim=0)  # 计算标准差

# 定义正常样本的边界（使用 3 倍标准差）
lower_bound = mu_overall - 3 * std_overall
upper_bound = mu_overall + 3 * std_overall

# Step 10: 在验证集上进行异常检测
valid_distances = torch.norm(X_valid_latent - mu_overall, dim=1)  # 计算验证集样本与均值的距离
valid_anomalies = (valid_distances > torch.norm(upper_bound - mu_overall)).float()  # 判断是否在边界外

print("Validation Set Anomalies:", valid_anomalies.sum())

# 在测试集上进行相似的检测
test_distances = torch.norm(X_test_latent - mu_overall, dim=1)
test_anomalies = (test_distances > torch.norm(upper_bound - mu_overall)).float()

print("Test Set Anomalies:", test_anomalies.sum())

# 计算验证集和测试集的性能（例如分类报告）
print("Validation Set Performance:")
# print(classification_report(y_valid, valid_anomalies.numpy(), target_names=["Normal", "Anomaly"]))
output_report(y_valid, valid_anomalies)

print("Test Set Performance:")
# print(classification_report(y_test, test_anomalies.numpy(), target_names=["Normal", "Anomaly"]))
output_report(y_test, test_anomalies)
