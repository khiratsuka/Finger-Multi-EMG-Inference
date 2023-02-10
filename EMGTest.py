# coding: utf-8
import argparse
import os

import torch
import torch.nn as nn

import dataset
import model
import training
from settings import *

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
    parser.add_argument('-t', '--training_target', help='対象の選択, [finger, 7key, 4key]', type=str, choices=['finger', '7key', '4key'])

    args = parser.parse_args()
    return args


def main():
    args = EMGTestArgparse()

    # 保存先のフォルダがなければ作る
    if not os.path.exists(args.csv_path):
        os.makedirs(args.csv_path)

    # 予測結果出力ファイルの宣言
    detail_pred_file = os.path.splitext(os.path.basename(args.model_path))[0] + '_detail_pred.csv'
    detail_pred_path = os.path.join(args.csv_path, detail_pred_file)

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

    # 学習に使う損失関数の定義
    criterion = nn.CrossEntropyLoss()

    # テストフェーズ
    _, test_acc, detail_pred = training.val_test(net, 'test', test_dataloader, 0, criterion, device, isDetailOutput=True)
    print('test_acc = {}'.format(test_acc/len(test_dataset)))

    # 予測結果と正解をcsvに出力する
    with open(detail_pred_path, 'w') as f:
        f.writelines(detail_pred)
    print('complete output prediction-detail: {}'.format(detail_pred_path))


if __name__ == '__main__':
    main()
