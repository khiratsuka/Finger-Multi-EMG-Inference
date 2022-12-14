# coding: utf-8
import os
import datetime
import csv
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import training
import model
import dataset
from settings import *


#GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    start_time = datetime.datetime.now()
    
    #データセットと結果保存のフォルダの宣言と作成
    dataset_folder = './dataset/'
    result_folder = './result/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    #datasetの生成
    train_EMG_dataset = dataset.EMGDataset(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=True)
    test_EMG_dataset  = dataset.EMGDataset(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=False)
    
    #学習と検証に使うデータセットのサイズを指定
    train_dataset_size = int(0.8 * len(train_EMG_dataset))
    val_dataset_size = len(train_EMG_dataset) - train_dataset_size
    train_EMG_dataset, val_EMG_dataset = torch.utils.data.random_split(train_EMG_dataset, [train_dataset_size, val_dataset_size])

    #dataloaderの生成
    Train_DataLoader = DataLoader(train_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)
    Val_DataLoader   = DataLoader(val_EMG_dataset,
                                  batch_size=2,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)
    Test_DataLoader  = DataLoader(test_EMG_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言
    net = model.EMG_Inference_Model().to(device)

    #学習に使う損失とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=net.parameters(), lr=lr)

    #学習曲線用
    history = {
        'train_loss':[],
        'train_acc':[],
        'val_loss':[],
        'val_acc':[]
    }

    #学習曲線の描画用
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(14.0, 8.0))    #プロットエリアの設定
    loss_train_lines,  = axes[0].plot(0, 0, '-ok')  #一度描画してlinesを取得
    loss_val_lines,    = axes[0].plot(0, 0, '-or')  #一度描画してlinesを取得
    acc_train_lines,   = axes[1].plot(0, 0, '-ok')  #一度描画してlinesを取得
    acc_val_lines,     = axes[1].plot(0, 0, '-or')  #一度描画してlinesを取得
    axes[0].set_xlim(0, num_epochs+1)    #x軸を最大値をエポック数+1へ
    axes[1].set_xlim(0, num_epochs+1)    #x軸を最大値をエポック数+1へ
    axes[0].set_title("loss")       #タイトルの設定
    axes[1].set_title("accuracy")   #タイトルの設定
    ax_x_list = []

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

        #x軸の一覧リストに現在のエポックを追加
        ax_x_list.append(epoch)

        #y軸のmaxとminを求める
        loss_y_min = min(history['train_loss']) if min(history['train_loss']) > min(history['val_loss']) else min(history['val_loss'])
        acc_y_min = min(history['train_acc']) if min(history['train_acc']) > min(history['val_acc']) else min(history['val_acc'])
        loss_y_max = max(history['train_loss']) if max(history['train_loss']) > max(history['val_loss']) else max(history['val_loss'])
        acc_y_max = max(history['train_acc']) if max(history['train_acc']) > max(history['val_acc']) else max(history['val_acc'])

        #y軸を更新
        axes[0].set_ylim(loss_y_min-1, loss_y_max+1)
        axes[1].set_ylim(acc_y_min-1, acc_y_max+1)

        #学習曲線のリアルタイム描画
        loss_train_lines.set_data(ax_x_list, history['train_loss'])
        loss_val_lines.set_data(ax_x_list, history['val_loss'])
        acc_train_lines.set_data(ax_x_list, history['train_acc'])
        acc_val_lines.set_data(ax_x_list, history['val_acc'])
        plt.pause(0.000000000001)

        print('----{} epochs----'.format(epoch))
        print('train_loss : ' + str(history['train_loss'][epoch]))
        print('train_acc : ' + str(history['train_acc'][epoch]))
        print('val_loss : ' + str(history['val_loss'][epoch]))
        print('val_acc : ' + str(history['val_acc'][epoch]))
        print('\n')
    
    #テストフェーズ
    test_loss, test_acc = training.val_test(net, 'test', Test_DataLoader, epoch, criterion, device)
    print('test_acc = {}'.format(test_acc))

    metrics = ['loss', 'acc']
    training.output_learningcurve(history, metrics, result_folder)

    #モデルを保存するフォルダの用意
    model_folder = './model/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    #モデルの保存
    full_model_name = 'finger_multi_emg_full' + start_time.strftime("%Y_%m_%d_%H%M%S") + '.pth'
    dict_model_name = 'finger_multi_emg' + start_time.strftime("%Y_%m_%d_%H%M%S") + '.pth'
    torch.save(net, model_folder + full_model_name)
    torch.save(net.state_dict(), model_folder + dict_model_name)


if __name__ == '__main__':
    main()
