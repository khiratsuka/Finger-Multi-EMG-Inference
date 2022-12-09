# coding: utf-8
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import training

#GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#分類ラベルの定義
LABEL_NAMES_DICT = {'thumb':0, 'index':1, 'middle':2, 'ring':3, 'pinkie':4}
LABEL_ID = []
LABEL_NAMES = []
for key, val in LABEL_NAMES_DICT.items():
    LABEL_ID.append(val)
    LABEL_NAMES.append(key)


#モデルの定義
class EMG_Inference_Model(nn.Module):
    def __init__(self):
        super(EMG_Inference_Model, self).__init__()

        #入力層のサイズは、1秒あたりのデータ数と何秒取るかを決めてから実際に測定して決める
        #https://b.meso.tokyo/post/173610335934/stm32-nucleo-adc この辺参考になるかも
        self.input_size = 4096
        self.hidden_size = 2048
        self.num_classes = len(LABEL_ID)

        self.fc0 = nn.Linear(self.input_size, self.hidden_size)
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = self.fc0(x)
        x = self.relu0(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        return x


#データセットの定義
class EMGDataset(Dataset):
    def __init__(self, dataset_folder='./dataset',
                 class_name='hoge',
                 is_train=True):
        self.dataset_folder = dataset_folder
        self.class_name = class_name
        self.is_train = is_train

        self.data_path, self.correct_class = self._get_file_names()

    def __getitem__(self, idx):
        emg_data_path, correct_class = self.data_path[idx], self.correct_class[idx]
        max_data_sampling_num = 25000
        append_num_data = 0.5

        #データの読み込み
        emg_data = np.loadtxt(emg_data_path, delimiter='\n')
        emg_data = np.reshape(emg_data, (emg_data.shape[0], 1))

        #FFTの実行、負の周波数の分もそのまま使う(使わない場合はN/2する)
        emg_fft_data = np.fft.fft(emg_data)
        max_emg_val = np.max(emg_data)
        min_emg_val = np.min(emg_data)

        #振幅の正規化
        for i in range(len(emg_fft_data)):
            emg_fft_data = (emg_data[i] - min_emg_val) / (max_emg_val - min_emg_val)

        tensor_emg_data = torch.FloatTensor(emg_fft_data)
        return tensor_emg_data, correct_class

    def __len__(self):
        return len(self.emg_data_path)

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        emg_data_path, correct_class, temp_class = [], [], [0]

        #set directory path
        for cname in self.class_name:
            emg_data_dir = os.path.join(self.emg_data_folder, cname, phase)
            data_names = sorted([name for name in os.listdir(emg_data_dir) if name.endswith('csv')])

            #checking directory of csv data
            for data_name in data_names:
                data_name_path = os.path.join(emg_data_dir, data_name)
                if not os.path.join(data_name_path):
                    continue

                emg_data_path.append(data_name_path)
                correct_class.append(LABEL_NAMES_DICT[cname])

        assert len(emg_data_path) == len(correct_class), 'number of emg_data and class are not same.'
        return list(emg_data_path), list(correct_class)



def main():
    start_time = datetime.datetime.now()
    
    #データセットと結果保存のフォルダの宣言と作成
    dataset_folder = './dataset/'
    result_folder = './result/'
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    #学習のパラメータ
    batch_size = 2
    num_epochs = 1000
    lr = 0.01

    #datasetの生成
    train_EMG_dataset = EMGDataset(emg_data_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=True)
    test_EMG_dataset  = EMGDataset(emg_data_folder = dataset_folder,
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
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)
    Val_DataLoader   = DataLoader(val_EMG_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)
    Test_DataLoader  = DataLoader(test_EMG_dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言
    net = EMG_Inference_Model().to(device)

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
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False)    #プロットエリアの設定
    loss_train_lines,  = axes[0, 0].plot(0, 0, '-ok')  #一度描画してlinesを取得
    loss_val_lines,    = axes[0, 0].plot(0, 0, '-or')  #一度描画してlinesを取得
    acc_train_lines,   = axes[0, 1].plot(0, 0, '-ok')  #一度描画してlinesを取得
    acc_val_lines,     = axes[0, 1].plot(0, 0, '-or')  #一度描画してlinesを取得
    axes[0, 0].set_xlim(0, num_epochs+1)    #x軸を最大値をエポック数+1へ
    axes[0, 1].set_xlim(0, num_epochs+1)    #x軸を最大値をエポック数+1へ
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
        axes[0, 0].set_ylim(loss_y_min-1, loss_y_max+1)
        axes[0, 1].set_ylim(acc_y_min-1, acc_y_max+1)

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
