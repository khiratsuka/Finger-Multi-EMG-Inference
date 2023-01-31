# coding: utf-8
import torch
import torch.nn as nn
from torch.quantization import DeQuantStub, QuantStub

from settings import *


# LinearとReluをひとまとめにしたモジュール
class FullConnect_Relu(nn.Module):
    def __init__(self, input_size, output_size, is_relu=True):
        super(FullConnect_Relu, self).__init__()
        self.is_relu = is_relu

        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        if self.is_relu:
            x = self.relu(x)
        return x


# センサ1つ分のネットワーク
class FullConnect_Relu_Ch(nn.Module):
    def __init__(self, input_size):
        super(FullConnect_Relu_Ch, self).__init__()
        self.fc_relu_ch_in = FullConnect_Relu(input_size, int(input_size/2))
        self.fc_ch_out = FullConnect_Relu(int(input_size/2), input_size, is_relu=False)

    def forward(self, x):
        x = self.fc_relu_ch_in(x)
        x = self.fc_ch_out(x)
        return x


# 各センサのネットワークで得られたベクトルを結合して全結合層へ流すネットワーク
class FullConnect_Relu_All(nn.Module):
    def __init__(self, ch_input_size, output_size=len(LABEL_ID)):
        super(FullConnect_Relu_All, self).__init__()
        input_size = ch_input_size * 8

        self.fc_relu_all_in = FullConnect_Relu(input_size, int(input_size/2))
        self.fc_relu_all_out = FullConnect_Relu(int(input_size/2), output_size, is_relu=False)

    def forward(self, ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7):
        x = torch.cat((ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7), 1)
        x = self.fc_relu_all_in(x)
        x = self.fc_relu_all_out(x)
        return x


class EMG_Inference_Model_Linear(nn.Module):
    def __init__(self, input_size, num_classes=len(LABEL_ID), is_transfer_train=False):
        super(EMG_Inference_Model_Linear, self).__init__()

        # 入力層のサイズは、1秒あたりのデータ数と何秒取るかを決めてから実際に測定して決める
        # https://b.meso.tokyo/post/173610335934/stm32-nucleo-adc この辺参考になるかも
        self.input_size = input_size
        self.num_classes = num_classes
        if is_transfer_train:
            self.num_classes = len(LABEL_ID_FINGER)

        self.quant = QuantStub()
        self.fc_relu_ch0 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch1 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch2 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch3 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch4 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch5 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch6 = FullConnect_Relu_Ch(self.input_size)
        self.fc_relu_ch7 = FullConnect_Relu_Ch(self.input_size)

        self.fc_relu_all = FullConnect_Relu_All(self.input_size, output_size=self.num_classes)
        self.dequant = DeQuantStub()

    def forward(self, data):
        data = self.quant(data)
        ch0 = self.fc_relu_ch0(data[0])
        ch1 = self.fc_relu_ch1(data[1])
        ch2 = self.fc_relu_ch2(data[2])
        ch3 = self.fc_relu_ch3(data[3])
        ch4 = self.fc_relu_ch4(data[4])
        ch5 = self.fc_relu_ch5(data[5])
        ch6 = self.fc_relu_ch6(data[6])
        ch7 = self.fc_relu_ch7(data[7])
        all_ch = self.fc_relu_all(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7)
        all_ch = self.dequant(all_ch)
        return all_ch

    def fuse_model(self):
        for m in self.modules():
            if type(m) == FullConnect_Relu_Ch:
                torch.quantization.fuse_modules(m.fc_relu_ch_in, ['fc', 'relu'], inplace=True)
            elif type(m) == FullConnect_Relu_All:
                torch.quantization.fuse_modules(m.fc_relu_all_in, ['fc', 'relu'], inplace=True)


# LSTM
class EMG_Inference_Model_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, label_size=len(LABEL_ID), batch_first=False):
        super(EMG_Inference_Model_LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.label_size = label_size
        self.batch_first = batch_first

        self.quant = QuantStub()
        self.lstm_layer = nn.LSTM(input_size=self.input_size,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.num_layers,
                                  batch_first=self.batch_first)
        self.fc_layer = nn.Linear(self.hidden_size, self.label_size)
        self.dequant = DeQuantStub()

    def forward(self, data):
        data = self.quant(data)
        data = torch.reshape(data, (data.size(2), data.size(1), data.size(0)))
        output, (last_hidden_out, last_cell_out) = self.lstm_layer(data, None)
        last_hidden_out = last_hidden_out[self.num_layers-1].view(-1, self.hidden_size)
        out = self.fc_layer(last_hidden_out)
        out = self.dequant(out)

        return out
