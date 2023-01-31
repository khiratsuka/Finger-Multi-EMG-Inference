# coding: utf-8
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import quantize_model
from settings import *

#GPUが使用可能であれば使う
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def main():              
    #データセットフォルダの宣言
    dataset_folder = './dataset_key/'

    #モデルファイルの宣言
    model_folder = './model/Key/'
    model_file = 'Key-FC-FFT'
    model_path = model_folder + model_file + '.pth'

    #データセットの生成 
    calib_EMG_dataset  = dataset.EMGDatasetFFT(dataset_folder = dataset_folder,
                                   class_name = LABEL_NAMES,
                                   is_train=True)

    #dataloaderの生成
    Calib_DataLoader  = DataLoader(calib_EMG_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=1,
                                  drop_last=True,
                                  pin_memory=True)

    #モデルの宣言と読み込み
    net = quantize_model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    #キャリブレーション
    torch.quantization.prepare(net, inplace=True)
    calib_n = 100
    calib_now = 0
    with tqdm(total=100, unit='batch', desc='[calibration]')as pb:
        #勾配の計算をしない
        with torch.no_grad():
            for data, label in Calib_DataLoader:
                if calib_now > calib_n:
                    break
                #データとクラスのラベルを学習するデバイスに載せる
                data = data.to(device)
                data = torch.reshape(data, (data.size(1), data.size(0), data.size(2))).float()
                label = label.to(device)

                #順伝搬
                pred = net(data)
                calib_now += 1

                pb.update(1)
    
    #量子化
    torch.quantization.convert(net, inplace=True)
    torch.save(net.to('cpu').state_dict(), model_folder+model_file+'_quantized.pth')

if __name__ == '__main__':
    main()

