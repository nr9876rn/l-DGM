import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import logging
import time
from eval_output import out_log


logging.basicConfig(filename='AE_dist_performance.log', level=logging.INFO)
logging.info(f"Performance of AE_dist model of sampling frac 0.2")


# Step 1: 数据读取
file_path = r'E:\Academicfiles\jupyterfiles\Causal-subgraph\Dataset_OCGNN-main\ethereum.txt'
data = pd.read_csv(file_path, header=None)

# 提取用户ID、标签和数据值
user_ids = data[0]
labels = data[35]
features = data.iloc[:, 1:35]

# Step 2: 保持标签比例的15%采样
data_sampled = data.groupby(35, group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=42)).reset_index(drop=True)

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


# Step 5: 构建 Autoencoder 模型
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
        return decoded


# 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 6: 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader):.4f}")


# Step 7: 获取潜在空间表示
def get_latent_space_representation(model, data_tensor):
    with torch.no_grad():
        latent_representation = model.encoder(data_tensor)  # 仅使用编码器
    return latent_representation


# 获取训练集和验证集的潜在空间表示
X_train_latent = get_latent_space_representation(model, X_train_tensor)
X_valid_latent = get_latent_space_representation(model, X_valid_tensor)
X_test_latent = get_latent_space_representation(model, X_test_tensor)

# Step 8: 计算正常样本的特征中心 (μ)
# 使用训练集标签 (y_train) 来选择正常样本
normal_train_latent = X_train_latent  # 仅使用标签为0的正常样本
μ = normal_train_latent.mean(dim=0)  # 特征中心 μ

# Step 9: 计算样本与特征中心的距离并定义阈值
normal_train_distances = torch.norm(normal_train_latent - μ, dim=1)

thresholds = [0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 1.00]

for p in thresholds:
    performance_output = f"Performance for threshold={p}:\n"
    logging.info(performance_output)

    threshold_distance = torch.quantile(normal_train_distances, p).item()
    print("threshold_distance:", threshold_distance)
    logging.info(f"threshold_distance =: {threshold_distance:.6f}")

    # Step 10: 在验证集上进行异常检测
    valid_distances = torch.norm(X_valid_latent - μ, dim=1)
    valid_predictions = (valid_distances > threshold_distance).float()

    print("Validation Set Performance:")
    print(classification_report(y_valid, valid_predictions.numpy(), target_names=["Normal", "Anomaly"]))

    # Step 11: 在测试集上进行异常检测
    start_test_time = time.time()  # 记录测试开始时间

    X_test_latent = get_latent_space_representation(model, X_test_tensor)
    test_distances = torch.norm(X_test_latent - μ, dim=1)
    test_predictions = (test_distances > threshold_distance).float()

    end_test_time = time.time()  # 记录测试结束时间
    test_time = end_test_time - start_test_time  # 计算测试时长
    logging.info(f"Testing time: {test_time:.6f} seconds")

    # print("Test Set Performance:")
    out_log(y_test, test_predictions)

    # print("Test Set Performance:")
    # print(classification_report(y_test, test_predictions.numpy(), target_names=["Normal", "Anomaly"]))
