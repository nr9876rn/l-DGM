import os
import pickle
import time
import numpy as np
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest as IForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve
)
from normalize import *
from utils import *
from src.data_load import load_data

SEED = 42
np.random.seed(SEED)

# FPR_list = np.arange(0.01, 0.51, 0.01)
# for 'cse_improved': ['server1']
# FPR_list = np.arange(0.38, 0.50, 0.04)
FPR_list = np.arange(0.38, 0.50, 0.04)



def generate_outliers(X_normal, noise_factor=1.6):
    """RODEO 伪离群点生成：对正常样本加噪声"""
    noise = np.random.normal(0, noise_factor, X_normal.shape)
    return X_normal + noise

def train_process(dataset, subset):
    X_train, X_eval, y_train, y_eval = load_data(dataset, subset, mode='train')
    X_train = X_train[y_train == 0]

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train, X_eval = normalizer.transform(X_train), normalizer.transform(X_eval)

    # 生成伪离群点
    pseudo_outliers = generate_outliers(X_train)
    X_comb = np.vstack([X_train, pseudo_outliers])
    y_comb = np.concatenate([np.zeros(len(X_train)), np.ones(len(pseudo_outliers))])

    best_model, best_score = None, 0
    best_time = 0

    for FPR in FPR_list:
        start_time = time.time()
        model = IForest(n_estimators=500, contamination=FPR, random_state=SEED)
        model.fit(X_comb)
        end_time = time.time()
        train_duration = end_time - start_time

        y_pred = model.predict(X_eval)
        y_pred[y_pred == 1] = 0
        y_pred[y_pred == -1] = 1
        score = f1_score(y_eval, y_pred)
        if score > best_score:
            print('contamination:', FPR, 'F1:', score)
            best_model = model
            best_score = score
            best_time = train_duration

    with open(os.path.join(TARGET_MODEL_DIR, f'RODEO_IF_{dataset}_{subset}.model'), 'wb') as f:
        pickle.dump(best_model, f)
    if not os.path.exists(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm')):
        with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'wb') as f:
            pickle.dump(normalizer, f)

    print(f"Best training time: {best_time:.4f} seconds")

def test_process(dataset, subset):
    with open(os.path.join(TARGET_MODEL_DIR, f'RODEO_IF_{dataset}_{subset}.model'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(NORMALIZER_DIR, f'{dataset}_{subset}.norm'), 'rb') as f:
        normalizer = pickle.load(f)

    X, y = load_data(dataset, subset, mode='test')
    X = normalizer.transform(X)
    y[y != 0] = 1

    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()
    test_duration = end_time - start_time

    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    tp = TP(y, y_pred)
    fp = FP(y, y_pred)
    tn = TN(y, y_pred)
    fn = FN(y, y_pred)

    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    score = -model.score_samples(X)
    auc = roc_auc_score(y, score)
    aupr = average_precision_score(y, score)

    fprs, tprs, _ = roc_curve(y, score)
    fpr95_index = (tprs >= 0.95).argmax()
    fpr95 = fprs[fpr95_index]

    print(f"Test time: {test_duration:.4f} seconds")
    print(f"TPR: {tpr:.4f}")
    print(f"TNR: {tnr:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auc:.4f}")
    print(f"AUPR: {aupr:.4f}")
    print(f"FPR95: {fpr95:.4f}")



# load data
"""
'cicids_custom': ['Tuesday'],
'toniot_custom': ['ddos'], 
'cse_improved': ['server1'], 
'Graph': ['ethereum'], 
'BNaT': ['w1']  
"""


if __name__ == '__main__':
    dataset = 'toniot_custom'
    subset = 'ddos'
    start_time = time.time()
    train_process(dataset, subset)
    print(f"Training time: {time.time() - start_time:.4f} seconds")
    test_process(dataset, subset)
