from pyod.models.deep_svdd import DeepSVDD

from data_load import *
from utils import *
import time
from torch.utils.data import DataLoader, TensorDataset



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


# 加载已经保存的归一化器
with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
    normalizer = pickle.load(f)
X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
X_test = normalizer.transform(X_test)

start_train_time_blackbox = time.time()  # 记录测试开始时间
contamination = 0.1  # percentage of outliers
n_features = X_train.shape[1]  # 特征数量
use_ae = False  # hyperparameter for use ae architecture instead of simple NN
random_state = 10  # if C is set to None use random_state
clf = DeepSVDD(n_features=n_features, use_ae=use_ae, epochs=50, contamination=contamination, random_state=random_state)
clf.fit(X_train)
train_time_blackbox = time.time() - start_train_time_blackbox
print(f"Blackbox Model Training Time: {train_time_blackbox:.4f} seconds")

start_test_time_blackbox = time.time()  # 记录测试开始时间
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)
evaluate_predictions(y_test, y_test_pred)
test_time_blackbox = time.time() - start_test_time_blackbox  # 计算测试时间

print(f"Blackbox Model Testing Time: {test_time_blackbox:.4f} seconds")
