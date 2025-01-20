from data_load import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import logging
import time
from eval_output import out_log

# 确保日志文件夹存在
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)

# 配置日志系统
log_file_path = os.path.join(log_dir, 'AE_dist_loss_performance.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO)
logging.info(f"Performance of AE_dist_loss model of sampling frac 1")


# Step 1: 数据读取
file_path = r'C:\Users\nr\Desktop\graduation project\UAD-Rule-Extraction-main\dataset\Graph\ethereum.txt'
data = pd.read_csv(file_path, header=None)

# 提取用户ID、标签和数据值
user_ids = data[0]
labels = data[35]
features = data.iloc[:, 1:35]

# Step 2: 保持标签比例的15%采样
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
        return decoded, encoded  # 返回解码结果和潜在空间表示


# 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
model = Autoencoder(input_dim)
criterion = nn.MSELoss()  # 重构误差损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Step 6: 定义潜在空间距离损失
def latent_space_distance_loss(latent_representation):
    """计算潜在空间中样本与均值的距离损失"""
    mean_latent = latent_representation.mean(dim=0)  # 计算潜在空间的均值
    distance = torch.norm(latent_representation - mean_latent, dim=1)  # 计算每个样本的距离
    return distance.mean()  # 返回所有样本与均值的平均距离作为损失


# Step 7: 训练模型
num_epochs = 50
latent_loss_weight = 0.1

start_train_time = time.time()  # 记录训练开始时间

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    latent_loss = 0
    for batch in train_loader:
        inputs = batch[0]
        outputs, latent_representation = model(inputs)

        # 计算重构误差
        reconstruction_loss = criterion(outputs, inputs)

        # 计算潜在空间的距离损失
        distance_loss = latent_space_distance_loss(latent_representation)

        # 总损失 = 重构误差 + 潜在空间距离损失
        total_loss = reconstruction_loss + distance_loss  # 可调节系数0.01
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        train_loss += reconstruction_loss.item()
        latent_loss += distance_loss.item()

    print(
        f"Epoch {epoch + 1}/{num_epochs}, Reconstruction Loss: {train_loss / len(train_loader):.4f}, Latent Loss: {latent_loss / len(train_loader):.4f}")

end_train_time = time.time()  # 记录训练结束时间
train_time = end_train_time - start_train_time  # 计算训练时长
logging.info(f"Training time : {train_time:.6f} seconds")


# Step 8: 获取潜在空间表示
def get_latent_space_representation(model, data_tensor):
    with torch.no_grad():
        _, latent_representation = model(data_tensor)  # 仅获取潜在空间表示
    return latent_representation


# 获取训练集和验证集的潜在空间表示
X_train_latent = get_latent_space_representation(model, X_train_tensor)
X_valid_latent = get_latent_space_representation(model, X_valid_tensor)
X_test_latent = get_latent_space_representation(model, X_test_tensor)

# Step 9: 计算正常样本的特征中心 (μ)
normal_train_latent = X_train_latent  # 仅使用标签为0的正常样本
μ = normal_train_latent.mean(dim=0)  # 特征中心 μ

# Step 10: 计算样本与特征中心的距离并定义阈值
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
