# coding: utf-8
import argparse
import datetime
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim

from utils import dataset, model, training
from utils.settings import *

# GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def EMGTransferTrainArgparse():
    parser = argparse.ArgumentParser(description='筋電位波形転移学習用プログラム')

    # 出入力パス系
    parser.add_argument('base_model_path', help='転移元のモデル', type=str)
    parser.add_argument('-m', '--model_path', help='モデルの保存先パス', type=str, default='./model')
    parser.add_argument('-d', '--dataset_path', help='読み込むデータセットのパス', type=str, default='./dataset_finger')
    parser.add_argument('-g', '--graph_path', help='学習曲線グラフの保存先', type=str, default='./result')
    parser.add_argument('-c', '--csv_path', help='学習曲線csvデータの保存先', type=str, default='./learning_curve_csv')

    # 学習設定系
    parser.add_argument('-a', '--model_arch', help='モデルアーキテクチャを指定, [fc, lstm]', type=str, choices=['fc', 'lstm'])
    parser.add_argument('-p', '--preprocess_data', help='データの前処理を指定, [raw, fft]', type=str, choices=['raw', 'fft'])
    parser.add_argument('-t', '--training_target', help='学習する内容の選択, [finger, 7key, 4key]', type=str, choices=['finger', '7key', '4key'])
    parser.add_argument('-l', '--learning_rate', help='学習率', type=float, default=0.01)
    parser.add_argument('-e', '--epoch', help='エポック数', type=int, default=200)
    parser.add_argument('-b', '--batch_size', help='バッチサイズ', type=int, default=1)
    parser.add_argument('--needPlot', help='学習曲線を学習中に表示するか', type=bool, default=True)
    parser.add_argument('--hasDropout', help='Dropout層を追加する', action='store_true')

    args = parser.parse_args()
    return args


def main():
    start_time = datetime.datetime.now()
    args = EMGTransferTrainArgparse()

    # 保存先のフォルダがなければ作る
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.graph_path):
        os.makedirs(args.graph_path)
    if not os.path.exists(args.csv_path):
        os.makedirs(args.csv_path)

    # データセットの生成
    train_dataset, val_dataset, test_dataset = dataset.createDataset(dataset_path=args.dataset_path,
                                                                     training_target=args.training_target,
                                                                     preprocess=args.preprocess_data)
    # データローダーの生成
    train_dataloader, val_dataloader, test_dataloader = dataset.createDataLoader(train_dataset,
                                                                                 val_dataset,
                                                                                 test_dataset,
                                                                                 batch_size=args.batch_size)

    # モデルの宣言
    # 最初の宣言は転移元のモデルアーキテクチャ
    net = model.createModel(model_arch=args.model_arch, training_target='finger', device=device, hasDropout=args.hasDropout)

    # モデルの読み込み
    net.load_state_dict(torch.load(args.base_model_path))

    # 最後のLinear層の出力数を変更
    net.fc_relu_all.fc_relu_all_out.fc = nn.Linear(in_features=int(RAW_DATA_LENGTH*4), out_features=len(LABEL_ID[args.training_target]))
    net = net.to(device)

    # 転移学習で更新するパラメータ格納
    update_param_transfer_train = []
    update_param_transfer_train_name = ['fc_relu_ch0.fc_relu_ch_out.fc.weight', 'fc_relu_ch0.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch1.fc_relu_ch_out.fc.weight', 'fc_relu_ch1.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch2.fc_relu_ch_out.fc.weight', 'fc_relu_ch2.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch3.fc_relu_ch_out.fc.weight', 'fc_relu_ch3.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch4.fc_relu_ch_out.fc.weight', 'fc_relu_ch4.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch5.fc_relu_ch_out.fc.weight', 'fc_relu_ch5.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch6.fc_relu_ch_out.fc.weight', 'fc_relu_ch6.fc_relu_ch_out.fc.bias',
                                        'fc_relu_ch7.fc_relu_ch_out.fc.weight', 'fc_relu_ch7.fc_relu_ch_out.fc.bias',
                                        'fc_relu_all.fc_relu_all_in.fc.weight', 'fc_relu_all.fc_relu_all_in.fc.bias',
                                        'fc_relu_all.fc_relu_all_out.fc.weight', 'fc_relu_all.fc_relu_all_out.fc.bias']

    if is_transfer_train:
        for name, param in net.named_parameters():
            if name in update_param_transfer_train_name:
                param.requires_grad = True
                update_param_transfer_train.append(param)
            else:
                param.requires_grad = False

    # 学習に使う損失関数とオプティマイザの定義
    criterion = nn.CrossEntropyLoss()
    if is_transfer_train:
        optimizer = optim.SGD(params=update_param_transfer_train, lr=args.learning_rate)
    else:
        optimizer = optim.SGD(params=net.parameters(), lr=args.learning_rate)

    # 学習曲線用
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if args.needPlot:
        # 学習曲線の描画用
        fig, axes = plt.subplots(nrows=1, ncols=2, sharex=False, figsize=(14.0, 8.0))  # プロットエリアの設定
        loss_train_lines,  = axes[0].plot(0, 0, '-ok')  # 一度描画してlinesを取得
        loss_val_lines,    = axes[0].plot(0, 0, '-or')  # 一度描画してlinesを取得
        acc_train_lines,   = axes[1].plot(0, 0, '-ok')  # 一度描画してlinesを取得
        acc_val_lines,     = axes[1].plot(0, 0, '-or')  # 一度描画してlinesを取得
        axes[0].set_xlim(0, args.epoch+1)  # x軸を最大値をエポック数+1へ
        axes[1].set_xlim(0, args.epoch+1)  # x軸を最大値をエポック数+1へ
        axes[0].set_title("loss")  # タイトルの設定
        axes[1].set_title("accuracy")  # タイトルの設定
        ax_x_list = []

    print('train start.\n')
    for epoch in range(args.epoch):
        # 学習フェーズ
        train_loss, train_acc = training.train(net, train_dataloader, epoch, criterion, optimizer, device)
        history['train_loss'].append(train_loss/len(train_dataset))
        history['train_acc'].append(train_acc/len(train_dataset))

        # 検証フェーズ
        val_loss, val_acc = training.val_test(net, 'val', val_dataloader, epoch, criterion, device)
        history['val_loss'].append(val_loss/len(val_dataset))
        history['val_acc'].append(val_acc/len(val_dataset))

        if args.needPlot:
            # x軸の一覧リストに現在のエポックを追加
            ax_x_list.append(epoch)

            # y軸のmaxとminを求める
            loss_y_min = min(history['train_loss']) if min(history['train_loss']) > min(history['val_loss']) else min(history['val_loss'])
            acc_y_min = min(history['train_acc']) if min(history['train_acc']) > min(history['val_acc']) else min(history['val_acc'])
            loss_y_max = max(history['train_loss']) if max(history['train_loss']) > max(history['val_loss']) else max(history['val_loss'])
            acc_y_max = max(history['train_acc']) if max(history['train_acc']) > max(history['val_acc']) else max(history['val_acc'])

            # y軸を更新
            axes[0].set_ylim(loss_y_min-1, loss_y_max+1)
            axes[1].set_ylim(acc_y_min-1, acc_y_max+1)

            # 学習曲線のリアルタイム描画
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

        # 50epochごとにモデルの保存
        if (epoch + 1) % 50 == 0:
            net.eval()
            net = net.to('cpu')

            # モデル種類を記録
            model_type = args.training_target + '_' + args.model_arch + '_' + args.preprocess_data + '_lr' + str(args.learning_rate) + '_batchsize' + str(args.batch_size)
            if args.hasDropout:
                model_type += '_dropout'

            # モデル名確定
            model_name = model_type + '_emg_' + start_time.strftime("%Y_%m_%d_%H%M%S") + '_' + str(epoch) + 'epoch' + '.pth'

            # モデル保存
            save_path = os.path.join(args.model_path, model_name)
            torch.save(net.state_dict(), save_path)

            net = net.to(device)

    # テストフェーズ
    _, test_acc = training.val_test(net, 'test', test_dataloader, epoch, criterion, device)
    print('test_acc = {}'.format(test_acc/len(test_dataset)))

    metrics = ['loss', 'acc']
    training.outputLearningCurve(history, metrics, start_time)
    training.outputLearningCurveValue(history, start_time)


if __name__ == '__main__':
    main()
