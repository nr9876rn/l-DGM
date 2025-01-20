import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from IPy import IP
from global_var import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle


def load_data(dataset, subset, mode='train', **kwargs):
    if dataset == 'cicids':
        X, y = load_cicids(subset)
    elif dataset == 'unsw':
        X, y = load_unsw(subset)
    elif dataset == 'cicids_custom':
        X, y = load_cicids_custom(subset)
    elif dataset == 'toniot_custom':
        X, y = load_toniot_custom(subset)
    elif dataset == 'cicids_improved':
        X, y = load_cicids_improved(subset, **kwargs)
    elif dataset == 'cse_improved':
        X, y = load_cse_improved(subset, **kwargs)
    # XGB数据集
    elif dataset == 'XGB':
        X, y = load_XGB(subset)
    # Graph数据集
    elif dataset == 'Graph':
        X, y = load_Graph(subset)
    elif dataset == 'Elliptic':
        X, y = load_Elliptic(subset)
    elif dataset == 'BNaT':
        X, y = load_BNaT(subset)
    else:
        print('no such dataset')
        exit()
    # test_size = 0.2 表示测试集占20%的数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    # mode == 'train'时，将训练集拆分为真正的训练集和验证集，验证集占25%
    if mode == 'train':
        X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, test_size=0.25, random_state=SEED)
        # 仅保留 y_train 中标签为 0 的样本，适用于只用正常样本进行训练的异常检测场景。
        X_train, y_train = X_train[y_train == 0], y_train[y_train == 0]
        if 'random_select' in kwargs:
            X_train = X_train[y_train == 0]
            idx_rand = np.random.randint(0, X_train.shape[0], kwargs['random_select'])
            X_train, y_train = X_train[idx_rand], y_train[idx_rand]

        return X_train, X_eval, y_train.astype(int), y_eval.astype(int)
    elif mode == 'test':
        return X_test, y_test


# CICIDS
# 创建编码映射，将文本标签转换为整数编码
def encode_label_cicids(col: pd.Series):
    all_labels = list(set(col))
    a2l, l2a = {'BENIGN': 0}, {0: 'BENIGN'}
    all_labels.remove('BENIGN')
    for i, att in enumerate(all_labels):
        a2l[att] = i + 1
        l2a[i + 1] = att
    return a2l, l2a

# 这个没有用上
def load_cicids(subset):
    df = pd.read_csv(os.path.join(CICIDS_DIR, CICIDS_DICT[subset] + '.pcap_ISCX.csv'))

    df = df[CICIDS_IP_COLS + CICIDS_FEAT_COLS + [CICIDS_LABEL_COL]]
    df.dropna(how='any', inplace=True)
    # df.drop(df[df.sum(axis=1) == np.inf].index, inplace=True)

    # only include two web servers' external comms
    # cond = df[' Source IP'].isin(CICIDS_SERVER_IPS) | df[' Destination IP'].isin(CICIDS_SERVER_IPS)
    cond = df[' Destination IP'].isin(CICIDS_SERVER_IPS)
    df = df[cond | (df[CICIDS_LABEL_COL] != 'BENIGN')]
    # cond = (df[' Source IP'].str.startswith('192.168.10') & df[' Destination IP'].str.startswith('192.168.10'))
    # df = df[(~cond) | (df[CICIDS_LABEL_COL] != 'BENIGN')]

    X = df[CICIDS_FEAT_COLS].to_numpy()
    a2l, l2a = encode_label_cicids(df[CICIDS_LABEL_COL])
    y = df[CICIDS_LABEL_COL].apply(lambda x: a2l[x]).to_numpy()

    return X, y

# 实际上CICIDS每次用的是这个
def load_cicids_custom(subset):
    # df = pd.read_csv(os.path.join(CUSTOM_DATA_DIR, 'CICIDS-2017', f'{subset}.csv'))
    # 路径加了一个点
    df = pd.read_csv(os.path.join('../dataset', 'CICIDS-2017', f'{subset}.csv'))

    # only include two web servers' external comms
    # 只保留的目标ip，和标签异常的数据（但其实所有数据标签都是正常的，所有是只保留了目标ip的）
    cond = df['dest-ip'].isin(CICIDS_SERVER_IPS)
    df = df[cond | (df[CUSTOM_LABEL_COL] != 0)]
    print(len(df))

    X = df[CUSTOM_FEAT_COLS].to_numpy()
    y = df[CUSTOM_LABEL_COL].to_numpy()

    return X, y

# CICIDS-improved
# 这个没有用上
def load_cicids_improved(subset, **kwargs):
    subset = str(subset).lower()
    df = pd.read_csv(os.path.join(CICIDS_2_DIR, subset + '.csv'))
    try:
        feat_size = kwargs['feat_size']
        # print(f'feat_size: {feat_size}')
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_ALL_COLS[:feat_size] + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_ALL_COLS[:feat_size]
    except:
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_COLS + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_COLS

    # filter attempted
    df = df[df[CICIDS_2_ATTEMPT_COL] == -1]

    # only include 3 servers' external comms
    cond = df[CICIDS_2_IP_COLS[1]].isin(CICIDS_2_SERVER_IPS)
    # cond = df[CICIDS_2_IP_COLS[0]].isin(CICIDS_2_CLIENT_IPS)
    df = df[cond | (df[CICIDS_2_LABEL_COL] != 'BENIGN')]

    X = df[columns_to_extract].to_numpy()
    y = df[CICIDS_2_LABEL_COL].apply(lambda x: x != 'BENIGN').astype('int').to_numpy()

    return X, y

# CSE-CICIDS-2018-improved
def load_cse_improved(subset, **kwargs):
    subset = str(subset).lower()
    df = pd.read_csv(os.path.join(CSE_DIR, subset + '.csv'))
    # 如果传入了 feat_size，从所有特征列中仅选择前 feat_size个特征
    try:
        feat_size = kwargs['feat_size']
        # print(f'feat_size: {feat_size}')
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_ALL_COLS[:feat_size] + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_ALL_COLS[:feat_size]
    except:
        df = df[CICIDS_2_IP_COLS + CICIDS_2_FEAT_COLS + [CICIDS_2_LABEL_COL, CICIDS_2_ATTEMPT_COL]]
        columns_to_extract = CICIDS_2_FEAT_COLS

    # filter attempted
    df = df[df[CICIDS_2_ATTEMPT_COL] == -1]

    # only include 2 servers' external comms
    # 保留特定ip的，或者异常数据
    cond = df[CICIDS_2_IP_COLS[1]].isin(CSE_SERVER_IPS)
    # cond = df[CICIDS_2_IP_COLS[0]].isin(CICIDS_2_CLIENT_IPS)
    df = df[cond | (df[CICIDS_2_LABEL_COL] != 'BENIGN')]

    X = df[columns_to_extract].to_numpy()
    y = df[CICIDS_2_LABEL_COL].apply(lambda x: x != 'BENIGN').astype('int').to_numpy()

    return X, y

# UNSW-NB15
# 没有用实际
def encode_label_unsw(col: pd.Series):
    all_labels = list(set(col))
    a2l, l2a = {'Normal': 0}, {0: 'Normal'}
    all_labels.remove('Normal')
    for i, att in enumerate(all_labels):
        a2l[att] = i + 1
        l2a[i + 1] = att
    return a2l, l2a

# 没有用实际
def load_unsw(subset):
    df = pd.read_csv(UNSW_DICT[subset])
    df.dropna(how='any', inplace=True)

    df = df[(df['proto'] == 'tcp') | (df['proto'] == 'udp')]

    X = df[UNSW_FEAT_COLS].to_numpy()
    # a2l, l2a = encode_label_unsw(df[UNSW_CAT_COL])
    # y = df[UNSW_CAT_COL].apply(lambda x: a2l[x]).to_numpy()
    y = df[UNSW_LABEL_COL].to_numpy()
    return X, y


def load_toniot_custom(subset):
    # df = pd.read_csv(os.path.join(CUSTOM_DATA_DIR, 'TON-IoT', f'{subset}.csv'))
    # 路径加了一个点
    df = pd.read_csv(os.path.join('../dataset', 'TON-IoT', f'{subset}.csv'))

    # cond = df['src_ip'].isin(TONIOT_SERVER_IPS) | df['dst_ip'].isin(TONIOT_SERVER_IPS) 
    # df = df[cond | (df[CUSTOM_LABEL_COL] != 0)]
    # cond1 = (df['dur'] > 0)
    # cond2 = df['dst_ip'].apply(lambda x: IP(x) < IP('224.0.0.0/4'))
    # df = df[cond1 & cond2]
    # cond3 = df['dst_ip'].str.startswith('192.168.1')
    # df = df[~cond3 | (df[CUSTOM_LABEL_COL] == 0)]
    
    df_att = df[df['label'] == 1]

    df_list = [df_att]
    # 路径加了一个点
    for f in os.listdir(os.path.join('../dataset', 'TON-IoT')):
        if f.startswith('normal'):
            # 路径加了一个点
            df_norm = pd.read_csv(os.path.join('../dataset', 'TON-IoT', f))#CUSTOM_DATA_DIR
            cond = df_norm['src_ip'].isin(['3.122.49.24']) | df_norm['dst_ip'].isin(['3.122.49.24']) # TONIOT_IPS
            df_norm = df_norm[cond]
            df_list.append(df_norm)
    df = pd.concat(df_list)

    X = df[CUSTOM_FEAT_COLS].to_numpy()
    y = df[CUSTOM_LABEL_COL].to_numpy()

    return X, y

# XGB
def load_XGB(subset):
    df = pd.read_csv(os.path.join('../dataset', 'XGB', f'{subset}.csv'))
    # 最后两行的字符串转换为数字
    # encoder = LabelEncoder()
    # df['ERC20_most_sent_token_type'] = encoder.fit_transform(df['ERC20_most_sent_token_type'])
    # df['ERC20_most_rec_token_type'] = encoder.fit_transform(df['ERC20_most_rec_token_type'])
    # 将空白字符串替换为 NaN
    df.replace(' ', np.nan, inplace=True)
    # 用0填充所有NaN值
    df = df.fillna(0)
    # 只保留异常数据
    # df = df[df['FLAG'] == 1]
    X = df[XGB_FEAT_COLS].to_numpy()
    y = df[XGB_LABEL_COL].to_numpy()
    # 数据归一化
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    return X, y

# Graph
def load_Graph(subset):
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

    # 不单独做归一化了。在统一的模型训练测试时有做，多做一次效果不好
    # # Step 3: 对1-34列进行归一化
    # scaler = MinMaxScaler()
    # sampled_features = scaler.fit_transform(sampled_features)
    #
    # # 保存归一化器，以便后续使用
    # os.makedirs(NORMALIZER_DIR, exist_ok=True)
    # with open(os.path.join(NORMALIZER_DIR, 'ethereum_norm.pkl'), 'wb') as f:
    #     pickle.dump(scaler, f)

    # 转换特征和标签
    X = sampled_features
    y = sampled_labels.to_numpy()
    return X, y


# Elliptic
def load_Elliptic(subset):
    # 加载数据集
    classes = pd.read_csv(r'C:\Users\nr\Desktop\graduation project\UAD-Rule-Extraction-main\dataset\Elliptic\elliptic_txs_classes.csv', index_col='txId')  # Node labels
    features = pd.read_csv(r'C:\Users\nr\Desktop\graduation project\UAD-Rule-Extraction-main\dataset\Elliptic\elliptic_txs_features.csv', header=None, index_col=0)  # Node features

    # Combine classes and features
    data = pd.concat([classes, features], axis=1)

    # Feature indices
    feature_idx = [i + 2 for i in range(93 + 72)]
    # 是否只使用部分特征
    if True:
        feature_idx = feature_idx[:94]

    # Filter data
    if True:
        data = data[data["class"] != "unknown"]
    # 采样
    data = data.sample(frac=0.2, random_state=42)

    # Convert labels to numerical format
    class_converter = {"1": 1, "2": 0, "unknown": 2}
    data["class"] = data["class"].map(class_converter)

    # Extract features (X) and labels (y)
    X = data[feature_idx].values
    y = data["class"].values

    return X, y

#BNaT
def load_BNaT(subset):
    file_path = r'C:\Users\nr\Desktop\graduation project\UAD-Rule-Extraction-main\dataset\BNaT\w1.csv'
    data = pd.read_csv(file_path, header=None)
    # print(data.columns)
    # Convert labels to numerical format
    # print(data.head(9))  # 打印前 9 行数据

    class_converter = {"0": 0, "1": 2, "DoS": 1, "FoT": 3, "MitM": 4}
    data[19] = data[19].map(class_converter)
    # print(data.head(9))  # 打印前 9 行数据
    # Filter data
    if True:
        data = data[data[19] != 2]
        data = data[data[19] != 3]
        data = data[data[19] != 4]
    # print(data.head(9))  # 打印前 9 行数据
    data = data.fillna(0)
    # print(data.head(9))  # 打印前 9 行数据
    # print(data.loc[:, 0:18].dtypes)
    # print(data.loc[:, 0:18].applymap(lambda x: isinstance(x, str)).any())
    # print(data[3].unique())  # 查看第 3 列的所有唯一值

    X = data.loc[:, 0:18].apply(pd.to_numeric, errors='coerce').to_numpy()
    X = np.nan_to_num(X)
    # print(np.isnan(X).any())  # 如果为 True，表示有无法转换的值变成了 NaN
    y = data[19].to_numpy()
    # print(X.dtype)

    # print(X.shape)  # 查看 X 的维度
    # print(X.dtype)  # 应为 float64 或 int64
    # print(X[:5])  # 查看前 5 行数据
    # print(y[:5])  # 查看前 5 行数据
    return X, y