# 卿雨竹
# 开发时间：2025-01-04 15:34
# 导入所需库
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import pickle
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
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

dataset = 'Graph'
subset = 'ethereum'
X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')
with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
    normalizer = pickle.load(f)
X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
X_test = normalizer.transform(X_test)


# 生成伪离群点：对正常样本添加噪声
def generate_outliers(X_normal, noise_factor=1.6):
    """生成伪离群点：对正常样本添加噪声"""
    noise = np.random.normal(0, noise_factor, X_normal.shape)
    X_outliers = X_normal + noise
    return X_outliers

# 记录训练时间
start_train_time = time.time()
# 获取正常样本
X_normal = X_train[y_train == 0]

# 生成伪离群点
X_synthetic_outliers = generate_outliers(X_normal)
y_synthetic_outliers = np.ones(X_synthetic_outliers.shape[0])  # 伪离群点标签为1

# 将生成的伪离群点与正常样本合并
X_combined = np.vstack([X_train[y_train == 0], X_synthetic_outliers])
y_combined = np.concatenate([y_train[y_train == 0], y_synthetic_outliers])

# 打乱数据
X_combined, y_combined = shuffle(X_combined, y_combined, random_state=42)

# 使用随机森林进行训练
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_combined, y_combined)
end_train_time = time.time()
train_time = end_train_time - start_train_time
print(f"训练时间: {train_time:.4f}秒")

start_test_time = time.time()
# 评估模型
y_pred = model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 计算TPR, TNR, Precision, F1
TPR = tp / (tp + fn)
TNR = tn / (tn + fp)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

end_test_time = time.time()
test_time = end_test_time - start_test_time
print(f"测试时间: {test_time:.4f}秒")

print(f"TPR: {TPR:.4f}")
print(f"TNR: {TNR:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
