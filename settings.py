#分類ラベルの定義
LABEL_NAMES_DICT = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'pinkie':4}
LABEL_ID = []
LABEL_NAMES = []
for key, val in LABEL_NAMES_DICT.items():
    LABEL_ID.append(val)
    LABEL_NAMES.append(key)

CH_NUM = 8
RAW_DATA_LENGTH = 4100

batch_size = 1
num_epochs = 200
lr = 0.01
