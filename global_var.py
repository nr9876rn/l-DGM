import os

SEED = 9
DEVICE = 0

# 路径加了一个点
PROJ_ROOT = '..'

# CICIDS-2017
# 路径加了一个点
CICIDS_DIR = '../dataset/CICIDS-2017/csv'
CICIDS_DICT = {
    'Monday': 'Monday-WorkingHours',
    'Tuesday': 'Tuesday-WorkingHours',
    'Wednesday': 'Wednesday-WorkingHours',
    'Thursday': 'Thursday-WorkingHours',
    'Thursday-1': 'Thursday-WorkingHours-Morning-WebAttacks',
    'Thursday-2': 'Thursday-WorkingHours-Afternoon-Infilteration',
    'Friday': 'Friday-WorkingHours',
    'Friday-1': 'Friday-WorkingHours-Morning',
    'Friday-2': 'Friday-WorkingHours-Afternoon-PortScan',
    'Friday-3': 'Friday-WorkingHours-Afternoon-DDos'
}
CICIDS_IP_COLS = [
    ' Source IP', ' Destination IP'
]
CICIDS_FEAT_COLS = [
    ' Protocol', ' Destination Port', 
    ' Flow Duration', ' Total Fwd Packets',
    ' Total Backward Packets', 'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
    ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std', 'Bwd Packet Length Max',
    ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std',
    ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
    'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
    ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
    ' Bwd IAT Max', ' Bwd IAT Min',
    ' Fwd Header Length',
    ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
    ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
    ' Packet Length Std', ' Packet Length Variance',
]
CICIDS_LABEL_COL = ' Label'
CICIDS_SERVER_IPS = ['192.168.10.50', '192.168.10.51']

# 路径加了一个点
CICIDS_PCAP_DIR = '../dataset/CICIDS-2017/pcap'
# 路径加了一个点
CICIDS_TIME = '../dataset/CICIDS-2017/pcap/attack_time.csv'

# CICIDS2017-improved
# 路径加了一个点
CICIDS_2_DIR = '../dataset/CICIDS2017_improved'
CICIDS_2_IP_COLS = [
    'Src IP', 'Dst IP'
]
CICIDS_2_FEAT_ALL_COLS = [
    'Dst Port', 'Protocol',
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min',
    'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 
    'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Packet Length Min',
    'Packet Length Max', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 
    'Fwd RST Flags', 'Bwd RST Flags','FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWR Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
    'Average Packet Size', 'Fwd Segment Size Avg', 'Bwd Segment Size Avg',
    'Fwd Bytes/Bulk Avg', 'Fwd Packet/Bulk Avg', 'Fwd Bulk Rate Avg',
    'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg', 'Bwd Bulk Rate Avg',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'FWD Init Win Bytes', 'Bwd Init Win Bytes',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max',
    'Idle Min', 'ICMP Code', 'ICMP Type', 'Total TCP Flow Time'
]
CICIDS_2_FEAT_COLS = CICIDS_2_FEAT_ALL_COLS[:40]
CICIDS_2_LABEL_COL = 'Label'
CICIDS_2_ATTEMPT_COL = 'Attempted Category'
CICIDS_2_SERVER_IPS = [
    '192.168.10.3', 
    '192.168.10.50', 
    '192.168.10.51',
]
CICIDS_2_CLIENT_IPS = [
    '192.168.10.19',
    '192.168.10.17',
    '192.168.10.16',
    '192.168.10.12',
    '192.168.10.9',
    '192.168.10.5',
    '192.168.10.8',
    '192.168.10.14',
    '192.168.10.15',
    '192.168.10.25'
]

# 路径加了一个点
CSE_DIR = '../dataset/CSE-CICIDS-2018-improved'
CSE_SERVER_IPS = [
    '172.31.69.25', 
    '172.31.69.28', 
]

# UNSW-NB15
# 路径加了一个点
UNSW_DIR = '../dataset/UNSW-NB15/train_test'
UNSW_DICT = {
    'TRAIN': os.path.join(UNSW_DIR, 'UNSW_NB15_training-set.csv'),
    'TEST': os.path.join(UNSW_DIR, 'UNSW_NB15_testing-set.csv')
}
UNSW_FEAT_COLS = [
    'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
    'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
    'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports', 
]

UNSW_CAT_COL = 'attack_cat'
UNSW_LABEL_COL = 'label'

# TON-IoT
# 路径加了一个点
TON_DIR = '../dataset/TON-IoT/attack_time'
TONIOT_IPS = ['3.122.49.24']

# custom feature vector
CUSTOM_TUPLE_COLS = [
    'src-ip', 'dest-ip', 'sport', 'dport', 'proto'
]
CUSTOM_FEAT_COLS = [
    'count', 'fwd_count', 'bwd_count', 
    'ps_mean', 'ps_max', 'ps_min', 'ps_var',
    'ps_fwd_mean', 'ps_fwd_max', 'ps_fwd_min', 'ps_fwd_var',
    'ps_bwd_mean', 'ps_bwd_max', 'ps_bwd_min', 'ps_bwd_var',
    'iat_mean', 'iat_max', 'iat_min', 'iat_var',
    'iat_fwd_mean', 'iat_fwd_max', 'iat_fwd_min', 'iat_fwd_var',
    'iat_bwd_mean', 'iat_bwd_max', 'iat_bwd_min', 'iat_bwd_var',
    'dur', 'l4_proto', 'service_port'
]
CUSTOM_CAT_COLS = ['category', 'sub_cat']
CUSTOM_LABEL_COL = 'label'

CUSTOM_DATA_DIR = os.path.join(PROJ_ROOT, 'dataset')

# XGB的特征
# XGB_FEAT_COLS = ['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx',
#                  'Time_Diff_between_first_and_last_(Mins)', 'Sent_tnx', 'Received_Tnx',
#                  'Number_of_Created_Contracts', 'Unique_Received_From_Addresses',
#                  'Unique_Sent_To_Addresses', 'min_value_received', 'max_value_received',
#                  'avg_val_received', 'min_val_sent', 'max_val_sent', 'avg_val_sent',
#                  'min_value_sent_to_contract', 'max_val_sent_to_contract', 'avg_value_sent_to_contract',
#                  'total_transactions_(including_tnx_to_create_contract)', 'total_Ether_sent',
#                  'total_ether_received', 'total_ether_sent_contracts', 'total_ether_balance',
#                  'Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent',
#                  'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr',
#                  'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_contract_addr', 'ERC20_avg_time_between_sent_tnx',
#                  'ERC20_avg_time_between_rec_tnx', 'ERC20_avg_time_between_rec_2_tnx',
#                  'ERC20_avg_time_between_contract_tnx', 'ERC20_min_val_rec', 'ERC20_max_val_rec',
#                  'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent',
#                  'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract',
#                  'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name', 'ERC20_most_sent_token_type',
#                  'ERC20_most_rec_token_type']
XGB_FEAT_COLS = ['Avg_min_between_sent_tnx', 'Avg_min_between_received_tnx',
                 'Time_Diff_between_first_and_last_(Mins)', 'Sent_tnx', 'Received_Tnx',
                 'Number_of_Created_Contracts', 'Unique_Received_From_Addresses',
                 'Unique_Sent_To_Addresses', 'min_value_received', 'max_value_received',
                 'avg_val_received', 'min_val_sent', 'max_val_sent', 'avg_val_sent',
                 'min_value_sent_to_contract', 'max_val_sent_to_contract', 'avg_value_sent_to_contract',
                 'total_transactions_(including_tnx_to_create_contract)', 'total_Ether_sent',
                 'total_ether_received', 'total_ether_sent_contracts', 'total_ether_balance',
                 'Total_ERC20_tnxs', 'ERC20_total_Ether_received', 'ERC20_total_ether_sent',
                 'ERC20_total_Ether_sent_contract', 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_addr',
                 'ERC20_uniq_sent_addr', 'ERC20_uniq_rec_contract_addr', 'ERC20_avg_time_between_sent_tnx',
                 'ERC20_avg_time_between_rec_tnx', 'ERC20_avg_time_between_rec_2_tnx',
                 'ERC20_avg_time_between_contract_tnx', 'ERC20_min_val_rec', 'ERC20_max_val_rec',
                 'ERC20_avg_val_rec', 'ERC20_min_val_sent', 'ERC20_max_val_sent', 'ERC20_avg_val_sent',
                 'ERC20_min_val_sent_contract', 'ERC20_max_val_sent_contract', 'ERC20_avg_val_sent_contract']
# , 'ERC20_uniq_sent_token_name', 'ERC20_uniq_rec_token_name', 'ERC20_most_sent_token_type', 'ERC20_most_rec_token_type'
XGB_LABEL_COL = 'FLAG'


# model path
MODEL_DIR = os.path.join(PROJ_ROOT, 'model')
NORMALIZER_DIR = os.path.join(MODEL_DIR, 'normalizer')
TARGET_MODEL_DIR = os.path.join(MODEL_DIR, 'target_model')
RULE_MODEL_DIR = os.path.join(MODEL_DIR, 'explanation')
# result path
RESULT_DIR = os.path.join(PROJ_ROOT, 'result')