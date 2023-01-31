#分類ラベルの定義
LABEL_NAMES_DICT = {'Q':0, 'A':1, 'W':2, 'D':3, 'C':4, 'F':5, 'G':6}
#LABEL_NAMES_DICT = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'pinkie':4}
#LABEL_NAMES_DICT = {'A':0, 'W':1, 'D':2, 'F':3}
#LABEL_NAMES_DICT = {'A':0, 'W':1, 'C':2, 'G':3}

LABEL_ID = []
LABEL_NAMES = []
for key, val in LABEL_NAMES_DICT.items():
    LABEL_ID.append(val)
    LABEL_NAMES.append(key)


LABEL_NAMES_DICT_FINGER = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'pinkie':4}
LABEL_ID_FINGER= []
LABEL_NAMES_FINGER = []
for key, val in LABEL_NAMES_DICT_FINGER.items():
    LABEL_ID_FINGER.append(val)
    LABEL_NAMES_FINGER.append(key)



CH_NUM = 8
RAW_DATA_LENGTH = 4100

batch_size = 1
num_epochs = 300
SGD_lr = 0.005
Adam_lr = 0.001

is_transfer_train = False
