# 分類ラベルの定義
LABEL_NAMES_DICT_FINGER = {'thumb': 0, 'index': 1, 'middle': 2, 'ring': 3, 'pinkie': 4}
LABEL_NAMES_DICT_7KEY = {'Q': 0, 'A': 1, 'W': 2, 'D': 3, 'C': 4, 'F': 5, 'G': 6}
LABEL_NAMES_DICT_4KEY = {'A': 0, 'W': 1, 'C': 2, 'G': 3}

LABEL_ID_FINGER = []
LABEL_NAMES_FINGER = []
for key, val in LABEL_NAMES_DICT_FINGER.items():
    LABEL_ID_FINGER.append(val)
    LABEL_NAMES_FINGER.append(key)

LABEL_ID_7KEY = []
LABEL_NAMES_7KEY = []
for key, val in LABEL_NAMES_DICT_7KEY.items():
    LABEL_ID_7KEY.append(val)
    LABEL_NAMES_7KEY.append(key)

LABEL_ID_4KEY = []
LABEL_NAMES_4KEY = []
for key, val in LABEL_NAMES_DICT_4KEY.items():
    LABEL_ID_4KEY.append(val)
    LABEL_NAMES_4KEY.append(key)


LABEL_NAMES_DICT = {'finger': LABEL_NAMES_DICT_FINGER, '7key': LABEL_NAMES_DICT_7KEY, '4key': LABEL_NAMES_DICT_4KEY}
LABEL_ID = {'finger': LABEL_ID_FINGER, '7key': LABEL_ID_7KEY, '4key': LABEL_ID_4KEY}
LABEL_NAMES = {'finger': LABEL_NAMES_FINGER, '7key': LABEL_NAMES_7KEY, '4key': LABEL_NAMES_4KEY}

CH_NUM = 8
RAW_DATA_LENGTH = 4100

is_transfer_train = False
