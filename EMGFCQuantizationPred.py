# coding: utf-8
import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import dataset, quantize_model
from utils.settings import *

# GPUが使用可能であれば使う
device = 'cpu'


def EMGFCQuantizationArgparse():
    parser = argparse.ArgumentParser(description='FC量子化モデル推論プログラム')

    # 出入力パス系
    parser.add_argument('model_path', help='読み込むモデルのパス', type=str)
    parser.add_argument('-d', '--dataset_path', help='読み込むデータセットのパス', type=str, default='./dataset_key')

    # モデル設定系
    parser.add_argument('-p', '--preprocess_data', help='データの前処理を指定, [raw, fft]', type=str, choices=['raw', 'fft'], default='raw')
    parser.add_argument('-t', '--training_target', help='対象の選択, [finger, 7key, 4key]', type=str, choices=['finger', '7key', '4key'], default='7key')

    args = parser.parse_args()
    return args


def main():
    args = EMGFCQuantizationArgparse()

    # データセットの生成
    train_dataset, val_dataset, test_dataset = dataset.createDataset(dataset_path=args.dataset_path,
                                                                     training_target=args.training_target,
                                                                     preprocess=args.preprocess_data)
    # データローダーの生成
    test_dataloader = dataset.createDataLoader(train_dataset, val_dataset, test_dataset, batch_size=1, isOnlyTest=True)

    # モデルの宣言と読み込み
    net = quantize_model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH, num_classes=len(LABEL_ID[args.training_target]))
    net.eval()
    net.fuse_model()
    net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    # キャリブレーションと量子化
    torch.quantization.prepare(net, inplace=True)
    torch.quantization.convert(net, inplace=True)
    net.load_state_dict(torch.load(args.model_path))

    # ネットワークを評価モードにする
    net.eval()

    acc = 0.0
    all_predict_time = 0.0

    with tqdm(total=len(test_dataloader), unit='batch', desc='[test]')as pb:
        # 勾配の計算をしない
        with torch.no_grad():
            for data, label in test_dataloader:
                # データとクラスのラベルを学習するデバイスに載せる
                data = data.to(device)
                data = torch.reshape(data, (data.size(1), data.size(0), data.size(2))).float()
                label = label.to(device)

                # 順伝搬
                start_time = time.perf_counter()
                pred = net(data)
                end_time = time.perf_counter()

                # 正解数を求める
                softmax = nn.Softmax(dim=1)
                pred = softmax(pred)  # softmaxへ流して確率を求める
                pred_of_label = torch.argmax(pred, dim=1)  # 確率が1番大きいラベルを求める
                for ln in range(len(label)):
                    if pred_of_label[ln] == label[ln]:
                        acc += 1.0

                batch_time = end_time - start_time
                all_predict_time += batch_time

                pb.update(1)

    print('accuracy = {}'.format(acc/len(test_dataset)))
    print('predict time = {}[sec]'.format(all_predict_time))


if __name__ == '__main__':
    main()
