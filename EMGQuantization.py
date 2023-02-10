# coding: utf-8
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import quantize_model
from settings import *

device = 'cpu'


def EMGQuantizationArgparse():
    parser = argparse.ArgumentParser(description='筋電位波形テスト用プログラム')

    # 出入力パス系
    parser.add_argument('base_model_path', help='読み込むモデルのパス', type=str)
    parser.add_argument('-m', '--model_path', help='モデルの保存先パス', type=str, default='./model')
    parser.add_argument('-d', '--dataset_path', help='読み込むデータセットのパス', type=str, default='./dataset_finger')

    # モデル設定系
    parser.add_argument('-a', '--model_arch', help='モデルアーキテクチャを指定, [fc, lstm]', type=str, choices=['fc', 'lstm'])
    parser.add_argument('-p', '--preprocess_data', help='データの前処理を指定, [raw, fft]', type=str, choices=['raw', 'fft'])
    parser.add_argument('-t', '--training_target', help='対象の選択, [finger, 7key, 4key]', type=str, choices=['finger', '7key', '4key'])

    args = parser.parse_args()
    return args


def main():
    args = EMGQuantizationArgparse()

    # 保存先のフォルダがなければ作る
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # データセットの生成
    calib_dataset, temp0_dataset, temp1_dataset = dataset.createDataset(dataset_path=args.dataset_path,
                                                                        training_target=args.training_target,
                                                                        preprocess=args.preprocess_data)
    # データローダーの生成
    calib_dataloader, _, _ = dataset.createDataLoader(calib_dataset, temp0_dataset, temp1_dataset)

    # モデルの宣言と読み込み
    net = quantize_model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH).to(device)
    net.load_state_dict(torch.load(args.base_model_path))
    net.eval()
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    # キャリブレーション
    torch.quantization.prepare(net, inplace=True)
    calib_n = 100
    calib_now = 0
    with tqdm(total=100, unit='batch', desc='[calibration]')as pb:
        # 勾配の計算をしない
        with torch.no_grad():
            for data, label in calib_dataloader:
                if calib_now > calib_n:
                    break
                # データとクラスのラベルを学習するデバイスに載せる
                data = data.to(device)
                data = torch.reshape(data, (data.size(1), data.size(0), data.size(2))).float()
                label = label.to(device)

                # 順伝搬
                pred = net(data)
                calib_now += 1

                pb.update(1)

    # 量子化
    torch.quantization.convert(net, inplace=True)
    quantized_model_name = os.path.splitext(os.path.basename(args.model_path))[0] + '_quantized.pth'
    quantized_model_path = os.path.join(args.model_path, quantized_model_name)
    torch.save(net.to('cpu').state_dict(), quantized_model_path)


if __name__ == '__main__':
    main()
