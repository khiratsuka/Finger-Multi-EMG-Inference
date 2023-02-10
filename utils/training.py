# coding: utf-8
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.settings import *


def train(net, dataloader, epoch, criterion, optimizer, device):
    # ネットワークを学習モードにする
    net.train()

    # 学習曲線用
    epoch_loss = 0.0
    epoch_acc = 0.0

    with tqdm(total=len(dataloader), unit='batch', desc='[train] {}epoch'.format(epoch))as pb:
        for data, label in dataloader:
            # データとクラスのラベルを学習するデバイスに載せる
            data = data.to(device)
            data = torch.reshape(data, (data.size(1), data.size(0), data.size(2))).float()
            label = label.to(device)

            # 順伝搬
            optimizer.zero_grad()
            pred = net(data)

            # 損失を求める
            loss = criterion(pred, label)

            # 正解数を求める
            softmax = nn.Softmax(dim=1)
            pred = softmax(pred)  # softmaxへ流して確率を求める
            pred_of_label = torch.argmax(pred, dim=1)  # 確率が1番大きいラベルを求める
            for ln in range(len(label)):
                if pred_of_label[ln] == label[ln]:
                    epoch_acc += 1.0

            # lossを逆伝搬、パラメータの更新
            loss.backward()
            optimizer.step()

            # 1バッチ分のlossを加算
            epoch_loss += (float(loss.item()) + 1e-12) * float(data.size(0))

            pb.update(1)

    return epoch_loss, epoch_acc


def val_test(net, mode, dataloader, epoch, criterion, device, training_target, isDetailOutput=False):
    # ネットワークを評価モードにする
    net.eval()

    # 学習曲線用
    epoch_loss = 0.0
    epoch_acc = 0.0

    # 予測クラスと正解クラスの保存用
    pred_class_list = []
    correct_class_list = []

    with tqdm(total=len(dataloader), unit='batch', desc='[{}] {}epoch'.format(mode, epoch))as pb:
        # 勾配の計算をしない
        with torch.no_grad():
            for data, label in dataloader:
                # データとクラスのラベルを学習するデバイスに載せる
                data = data.to(device)
                data = torch.reshape(data, (data.size(1), data.size(0), data.size(2))).float()
                label = label.to(device)
                int_correct_label = int(label[0].to('cpu'))

                # 順伝搬
                pred = net(data)

                # 損失を求める
                loss = criterion(pred, label)

                # 正解数を求める
                softmax = nn.Softmax(dim=1)
                pred = softmax(pred)  # softmaxへ流して確率を求める
                pred_of_label = torch.argmax(pred, dim=1)  # 確率が1番大きいラベルを求める
                int_pred_of_label = int(pred_of_label[0].to('cpu'))
                for ln in range(len(label)):
                    if pred_of_label[ln] == label[ln]:
                        epoch_acc += 1.0

                # 予測・正解クラスのリスト作成
                pred_class = [k for k, v in LABEL_NAMES_DICT[training_target].items() if v == int_pred_of_label][0]
                correct_class = [k for k, v in LABEL_NAMES_DICT[training_target].items() if v == int_correct_label][0]
                pred_class_list.append(pred_class)
                correct_class_list.append(correct_class)

                # 1バッチ分のlossを加算
                epoch_loss += float(loss.item()) * float(data.size(0))

                pb.update(1)

    # 予測・正解クラスをcsvへ出力
    if isDetailOutput:
        output_data = ['correct_class, pred_class\n']
        for pred_c, correct_c in zip(pred_class_list, correct_class_list):
            output_line_data = correct_c + ', ' + pred_c + '\n'
            output_data.append(output_line_data)
        return epoch_loss, epoch_acc, output_data

    return epoch_loss, epoch_acc


def outputLearningCurve(data, metrics, start_time, result_folder='./result/'):
    plt.figure(figsize=(16, 9))
    for i in range(len(metrics)):
        metric = metrics[i]

        plt.subplot(1, 2, i+1)
        plt.title(metric, fontsize=25)
        plt_train = data['train_' + metric]
        plt_val = data['val_' + metric]
        plt.plot(plt_train, label='train')
        plt.plot(plt_val,   label='val')
        if metric == 'loss':
            plt.ylabel('loss', fontsize=23)
        elif metric == 'acc':
            plt.ylabel('accuracy', fontsize=23)
        plt.xlabel('epoch', fontsize=23)
        plt.legend(fontsize=20)
        plt.tick_params(labelsize=20)

    fname = start_time.strftime("%Y_%m_%d_%H%M%S") + '_eval.png'
    plt.savefig(result_folder+fname)


def outputLearningCurveValue(data, start_time, result_folder='./learning_curve_csv'):
    output_folder = start_time.strftime("%Y_%m_%d_%H%M%S")
    output_path = os.path.join(result_folder, output_folder)
    pathes = {}

    # それぞれのファイルのパス
    pathes['train_loss'] = os.path.join(output_path, 'train_loss.csv')
    pathes['train_acc'] = os.path.join(output_path, 'train_acc.csv')
    pathes['val_loss'] = os.path.join(output_path, 'val_loss.csv')
    pathes['val_acc'] = os.path.join(output_path, 'val_acc.csv')

    # 辞書形式になっているデータとパスを取り出して保存
    for val_data, path in zip(data.values(), pathes.values()):
        temp_val = []
        for val in val_data:
            temp_val.append(str(val) + '\n')
        with open(path, 'w') as f:
            f.writelines(temp_val)
