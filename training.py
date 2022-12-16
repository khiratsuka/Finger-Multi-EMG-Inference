# coding: utf-8
import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from settings import *

def train(net, dataloader, epoch, criterion, optimizer, device):
    #ネットワークを学習モードにする
    net.train()

    #学習曲線用
    epoch_loss = 0.0
    epoch_acc = 0.0

    with tqdm(total=len(dataloader), unit='batch', desc='[train] {}epoch'.format(epoch))as pb:
        for data, label in dataloader:
            #データとクラスのラベルを学習するデバイスに載せる
            data = data.to(device)
            #print(data.shape)
            #ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7 = data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]
            #ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7 = ch0.to(device), ch1.to(device), ch2.to(device), ch3.to(device), ch4.to(device), ch5.to(device), ch6.to(device), ch7.to(device)
            label = label.to(device)

            #順伝搬
            optimizer.zero_grad()
            pred = net(data)

            #損失を求める
            loss = criterion(pred, label)
            
            #正解数を求める
            softmax = nn.Softmax(dim=1)
            pred = softmax(pred)        #softmaxへ流して確率を求める
            pred_of_label = torch.argmax(pred, dim=1)   #確率が1番大きいラベルを求める
            for ln in range(len(label)):
                if pred_of_label[ln] == label[ln]:
                    epoch_acc += 1.0
            
            #lossを逆伝搬、パラメータの更新
            loss.backward()
            optimizer.step()

            #1バッチ分のlossを加算
            epoch_loss += float(loss.item()) * float(batch_size)

            pb.update(1)
    
    return epoch_loss, epoch_acc


def val_test(net, mode, dataloader, epoch, criterion, device):
    #ネットワークを評価モードにする
    net.eval()

    #学習曲線用
    epoch_loss = 0.0
    epoch_acc = 0.0

    with tqdm(total=len(dataloader), unit='batch', desc='[{}] {}epoch'.format(mode, epoch))as pb:
        #勾配の計算をしない
        with torch.no_grad():
            for data, label in dataloader:
                #データとクラスのラベルを学習するデバイスに載せる
                data = data.to(device)
                label = label.to(device)

                #順伝搬
                pred = net(data)

                #損失を求める
                loss = criterion(pred, label)
                
                #正解数を求める
                softmax = nn.Softmax(dim=1)
                pred = softmax(pred)        #softmaxへ流して確率を求める
                pred_of_label = torch.argmax(pred, dim=1)   #確率が1番大きいラベルを求める
                for ln in range(len(label)):
                    if pred_of_label[ln] == label[ln]:
                        epoch_acc += 1.0
                

                #1バッチ分のlossを加算
                epoch_loss += float(loss.item()) * float(data.size(0))

                pb.update(1)
    
    return epoch_loss, epoch_acc


def output_learningcurve(data, metrics, result_folder='./result/'):
    plt.figure(figsize=(10, 5))
    for i in range(len(metrics)):
        metric = metrics[i]

        plt.subplot(1, 2, i+1)
        plt.title(metric)
        plt_train = data['train_' + metric]
        plt_val   = data['val_'   + metric]
        plt.plot(plt_train, label='train')
        plt.plot(plt_val,   label='val')
        if metric  == 'loss':
            plt.ylabel('loss')
        elif metric == 'acc':
            plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
    now = datetime.datetime.now()
    fname = now.strftime("%Y_%m_%d_%H%M%S") + '_eval.png'
    plt.savefig(result_folder+fname)
