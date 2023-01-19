# coding: utf-8
import os

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
    dataset_folder = './dataset_full/'

    #モデルファイルの宣言
    model_folder = './model/Key-RAW-FC/'
    model_file = 'finger_multi_emg_2023_01_18_023027'
    model_path = model_folder + model_file + '.pth'

    #予測結果出力ファイルの宣言
    detail_pred_folder = './result/detail_pred'
    detail_pred_file = model_file + '_detail_pred.csv'
    detail_pred_path = os.path.join(detail_pred_folder, detail_pred_file)

    #デーセセットの生成 
    test_EMG_dataset  = dataset.EMGDatasetRawData(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=False)

    #テストに使うデータセットのサイズを指定
    test_dataset_size = len(test_EMG_dataset)

    #dataloaderの生成
    Test_DataLoader  = DataLoader(test_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言と読み込み
    net = model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH).to(device)
    net.load_state_dict(torch.load(model_path))
    #net = model.EMG_Inference_Model_LSTM(input_size=CH_NUM, hidden_size=int(RAW_DATA_LENGTH/4)).to(device)
    
    #学習に使う損失関数の定義
    criterion = nn.CrossEntropyLoss()

    #テストフェーズ
    test_loss, test_acc, detail_pred = training.val_test(net, 'test', Test_DataLoader, 0, criterion, device, isDetailOutput=True)
    print('test_acc = {}'.format(test_acc/test_dataset_size))
    
    #予測結果と正解をcsvに出力する
    if not os.path.exists(detail_pred_path):
        if not os.path.exists(detail_pred_folder):
            os.makedirs(detail_pred_folder)
        with open(detail_pred_path, 'w') as f:
            f.writelines(detail_pred)
        print('complete output prediction-detail: {}'.format(detail_pred_path))


if __name__ == '__main__':
    main()
