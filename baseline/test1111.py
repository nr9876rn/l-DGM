# 卿雨竹
# 开发时间：2024-11-11 19:09
# 换用XGBoost数据集
import sys
import numpy as np
import pickle
import torch
from global_var import *
from normalize import *
from data_load import *
from utils import *
from AE import AutoEncoder
from VAE import VAE
import KITree

torch.manual_seed(SEED)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load data
"""
available datasets and subsets: 
cicids_custom: [Tuesday, Wednesday, Thursday, Friday]
toniot_custom: [
    backdoor, ddos, dos,
    injection, mitm, password,
    runsomware, scanning, xss
]
cse_improved: [server1, server2]
XGB: [Complete, Complete_large]
"""

dataset = 'XGB'
subset = 'Complete'
X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
X_test, y_test = load_data(dataset, subset, mode='test')
with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
    normalizer = pickle.load(f)
X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)
X_test = normalizer.transform(X_test)

# load black box model
"""
available trained models: AE, VAE, OCSVM, IForest
"""

blackbox = 'AE'
try:
    model = torch.load(os.path.join(TARGET_MODEL_DIR, f'{blackbox}_{dataset}_{subset}.model'), map_location=DEVICE).to(DEVICE)
    thres = model.thres
except:
    with open(os.path.join(TARGET_MODEL_DIR, f'{blackbox}_{dataset}_{subset}.model'), 'rb') as f:
        model = pickle.load(f)
    thres = -model.offset_


# test well-trained blackbox model
y_origin = model.predict(X_test)
# sklearn models have different prediction format
if blackbox == 'OCSVM' or blackbox == 'IForest':
    y_origin[y_origin == 1] = 0
    y_origin[y_origin == -1] = 1

# 调用 evaluate_predictions() 函数评估模型的预测性能
evaluate_predictions(y_test, y_origin)


# prepare input of rule model
if blackbox == 'OCSVM' or blackbox == 'IForest':
    score = -model.score_samples(X_train)
    func = lambda x: -model.score_samples(x)
else:
    score = model.score_samples(X_train)
    func = lambda x: model.score_samples(x)

# extract rules from blackbox model using default hyperparameters
"""
here are some important hyperparameters you may try to calibrate:
max_levels = [5, 10, 15, 20, 25, 30]
n_beams = [2, 4, 6, 8, 10]
rhos = [0.01, 0.05, 0.1, 0.5, 1]
etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
"""
rule_model = KITree.KITree(func, thres)
rule_model.fit(X_train, score)

# predict() returns 0 if normal else 1
y_pred = rule_model.predict(X_test)
X_perturb = perturb_data_point(X_test)
y_perturb = rule_model.predict(X_perturb)

# 调用 evaluate_rule_model() 函数评估规则模型的预测效果
evaluate_rule_model(y_test, y_pred, y_origin, y_perturb)

# 获取规则树的深度
depth = rule_model.get_depth()
print(f"Rule Tree Depth: {depth}")

# 获取规则树的叶子节点数量
leaf_num = rule_model.get_leaf_num()
print(f"Number of Leaf Nodes: {leaf_num}")

# 获取规则字典
# 该函数返回从根到叶的路径（决策规则），如果所有维度的值都是 [-inf, -inf]，则表示该叶节点始终输出异常
# 输入的第一个参数 要根据谁的特征来变化
rules_dict = rule_model.get_rules_dict(XGB_FEAT_COLS, normalizer)
print(f"Extracted Rules:\n")
for rule_path, rule_content in rules_dict.items():
    print(f"Path: {rule_path}, Rule: {rule_content}")