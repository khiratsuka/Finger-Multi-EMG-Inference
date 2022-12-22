# coding: utf-8
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import model
import training
from settings import *

#GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():              
    #データセットフォルダの宣言
    dataset_folder = './dataset/'

    #モデルファイルの宣言
    model_path = './model/finger_multi_emg_2022_12_22_042446.pth'

    #デーセセットの生成 
    test_EMG_dataset  = dataset.EMGDatasetRawData(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=False)

    #テストに使うデータセットのサイズを指定
    test_dataset_size = len(test_EMG_dataset)

    #dataloaderの生成
    Test_DataLoader  = DataLoader(test_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言
    net = model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH).to(device)
    net.load_state_dict(torch.load(model_path))
    #net = model.EMG_Inference_Model_LSTM(input_size=CH_NUM, hidden_size=int(RAW_DATA_LENGTH/4)).to(device)

    #学習に使う損失関数の定義
    criterion = nn.CrossEntropyLoss()

    #テストフェーズ
    test_loss, test_acc = training.val_test(net, 'test', Test_DataLoader, 0, criterion, device)
    print('test_acc = {}'.format(test_acc/test_dataset_size))


if __name__ == '__main__':
    main()
