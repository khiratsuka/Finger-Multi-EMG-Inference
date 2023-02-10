# coding: utf-8
import argparse
import time

import torch
import torch.nn as nn
from tqdm import tqdm

from utils import dataset, model
from utils.settings import *

# GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def EMGTestArgparse():
    parser = argparse.ArgumentParser(description='筋電位波形テスト用プログラム')

    # 出入力パス系
    parser.add_argument('model_path', help='読み込むモデルのパス', type=str)
    parser.add_argument('-d', '--dataset_path', help='読み込むデータセットのパス', type=str, default='./dataset_finger')
    parser.add_argument('-c', '--csv_path', help='テスト結果のcsvデータの保存先', type=str, default='./result/detail_pred')

    # モデル設定系
    parser.add_argument('-a', '--model_arch', help='モデルアーキテクチャを指定, [fc, lstm]', type=str, choices=['fc', 'lstm'])
    parser.add_argument('-p', '--preprocess_data', help='データの前処理を指定, [raw, fft]', type=str, choices=['raw', 'fft'])
    parser.add_argument('-t', '--training_target', help='学習する内容の選択, [finger, 7key, 4key]', type=str, choices=['finger', '7key', '4key'])

    args = parser.parse_args()
    return args


def main():
    args = EMGTestArgparse()

    # データセットの生成
    test_dataset = dataset.createDataset(dataset_path=args.dataset_path,
                                         training_target=args.training_target,
                                         preprocess=args.preprocess_data,
                                         isOnlyTest=True)
    # データローダーの生成
    test_dataloader = dataset.createDataLoader(test_dataset=test_dataset,
                                               isOnlyTest=True)

    # モデルの宣言と読み込み
    net = model.createModel(model_arch=args.model_arch, training_target=args.training_target, device=device, hasDropout=args.hasDropout)
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
