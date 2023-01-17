#分類ラベルの定義
LABEL_NAMES_DICT = {'Q':0, 'A':1, 'W':2, 'D':3, 'C':4, 'F':5, 'G':6}
LABEL_ID = []
LABEL_NAMES = []
for key, val in LABEL_NAMES_DICT.items():
    LABEL_ID.append(val)
    LABEL_NAMES.append(key)

CH_NUM = 8
RAW_DATA_LENGTH = 4100

batch_size = 1
num_epochs = 300
lr = 0.005
