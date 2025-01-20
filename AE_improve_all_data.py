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
log_file_path = os.path.join(log_dir, 'AE_improve_performance_all_data.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO)
logging.info(f"Performance of AE_improve model of sampling frac 1")


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

dataset = 'cicids_custom'
subset = 'Tuesday'
logging.info(f"dataset : {dataset} ")
X_train, X_valid, y_train, y_valid = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')

# 创建MinMaxScaler对象，按列归一化数据
scaler = MinMaxScaler()

# 对训练集、验证集和测试集的特征进行归一化
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)  # 使用训练集的scaler进行变换
X_test = scaler.transform(X_test)  # 使用训练集的scaler进行变换

# # 将数据集合并为一个DataFrame，以便进行采样操作
# train_data = pd.DataFrame(X_train)
# train_data['label'] = y_train  # 添加标签列
#
# # Step 1: 按照标签分组，并在每个组中随机采样x%数据
# train_sampled = train_data.groupby('label', group_keys=False).apply(lambda x: x.sample(frac=1, random_state=42)).reset_index(drop=True)
#
# # 更新采样后的特征和标签
# X_train_sampled = train_sampled.iloc[:, :-1].values  # 选择所有特征列
# y_train_sampled = train_sampled['label'].values  # 选择标签列




# 转换为 Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=32, shuffle=True)


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
# Step 9: 计算样本与特征中心的距离并定义阈值
normal_train_distances = torch.norm(normal_train_latent - μ, dim=1)

# thresholds = [0.81, 0.82, 0.83, 0.84, 0.85, 0.86,0.87, 0.88, 0.89, 0.9, 0.91]
thresholds = [0.81, 0.82, 0.83, 0.84, 0.85, 0.86,0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 0.9999, 1.00]


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
