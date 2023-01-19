# coding: utf-8
import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import dataset
import model
import training
from settings import *

#GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    start_time = datetime.datetime.now()
    
    #データセットと結果保存のフォルダの宣言と作成
    dataset_folder = './dataset_full/'
    result_folder = './result/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    #datasetの生成
    train_EMG_dataset = dataset.EMGDatasetRawData(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=True)                     
    test_EMG_dataset  = dataset.EMGDatasetRawData(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=False)
    
    #学習と検証とテストに使うデータセットのサイズを指定
    train_dataset_size = int(0.938 * len(train_EMG_dataset))
    val_dataset_size = len(train_EMG_dataset) - train_dataset_size
    test_dataset_size = len(test_EMG_dataset)
    train_EMG_dataset, val_EMG_dataset = torch.utils.data.random_split(train_EMG_dataset, [train_dataset_size, val_dataset_size])

    #dataloaderの生成
    Train_DataLoader = DataLoader(train_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)
    Val_DataLoader   = DataLoader(val_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)
    Test_DataLoader  = DataLoader(test_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言
    net = model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH, is_transfer_train=True)
    #net = model.EMG_Inference_Model_LSTM(input_size=CH_NUM, hidden_size=int(RAW_DATA_LENGTH/4)).to(device)

    #モデルの読み込み
    model_path = './model/Finger-RAW-FC/finger_multi_emg_2022_12_22_075210.pth'
    net.load_state_dict(torch.load(model_path))

    #最後のLinear層の出力数を変更
    net.fc_relu_all.fc_relu_all_out.fc = nn.Linear(in_features=int(RAW_DATA_LENGTH*4), out_features=len(LABEL_ID))
    net = net.to(device)
    
    #転移学習で更新するパラメータ格納
    update_param = []
    update_param_name = ['fc_relu_ch0.fc_relu_ch_out.fc.weight', 'fc_relu_ch0.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch1.fc_relu_ch_out.fc.weight', 'fc_relu_ch1.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch2.fc_relu_ch_out.fc.weight', 'fc_relu_ch2.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch3.fc_relu_ch_out.fc.weight', 'fc_relu_ch3.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch4.fc_relu_ch_out.fc.weight', 'fc_relu_ch4.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch5.fc_relu_ch_out.fc.weight', 'fc_relu_ch5.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch6.fc_relu_ch_out.fc.weight', 'fc_relu_ch6.fc_relu_ch_out.fc.bias',
                         'fc_relu_ch7.fc_relu_ch_out.fc.weight', 'fc_relu_ch7.fc_relu_ch_out.fc.bias',
                         'fc_relu_all.fc_relu_all_out.fc.weight', 'fc_relu_all.fc_relu_all_out.fc.bias']

    for name, param in net.named_parameters():
        if name in update_param_name:
            param.requires_grad = True
            update_param.append(param)
        else:
            param.requires_grad = False

    #学習に使う損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=update_param, lr=SGD_lr)

    #学習曲線用
    history = {
        'train_loss':[],
        'train_acc':[],
        'val_loss':[],
        'val_acc':[]
    }

    #モデルを保存するフォルダの用意
    model_folder = './model/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    print('train start.\n')
    for epoch in range(num_epochs):
        #学習フェーズ
        train_loss, train_acc = training.train(net, Train_DataLoader, epoch, criterion, optimizer, device)
        history['train_loss'].append(train_loss/train_dataset_size)
        history['train_acc'].append(train_acc/train_dataset_size)

        #検証フェーズ
        val_loss, val_acc = training.val_test(net, 'val', Val_DataLoader, epoch, criterion, device)
        history['val_loss'].append(val_loss/val_dataset_size)
        history['val_acc'].append(val_acc/val_dataset_size)

        print('----{} epochs----'.format(epoch))
        print('train_loss : ' + str(history['train_loss'][epoch]))
        print('train_acc : ' + str(history['train_acc'][epoch]))
        print('val_loss : ' + str(history['val_loss'][epoch]))
        print('val_acc : ' + str(history['val_acc'][epoch]))
        print('\n')

        #モデルの保存
        if (epoch + 1) % 50 == 0:
            net.eval()
            net = net.to('cpu')
            dict_model_name = 'finger_multi_emg_' + start_time.strftime("%Y_%m_%d_%H%M%S") + ' _' + str(epoch) + 'epoch' + '.pth'
            torch.save(net.state_dict(), model_folder + dict_model_name)
            net = net.to(device)
        
    #テストフェーズ
    test_loss, test_acc = training.val_test(net, 'test', Test_DataLoader, 0, criterion, device)
    print('test_acc = {}'.format(test_acc/test_dataset_size))

    metrics = ['loss', 'acc']
    training.outputLearningCurve(history, metrics, start_time)
    training.outputLearningCurveValue(history, start_time)






if __name__ == '__main__':
    main()
