# coding: utf-8
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import dataset
import model
from settings import *

# GPUが使用可能であれば使う
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # データセットフォルダの宣言
    dataset_folder = './dataset_key/'

    # モデルファイルの宣言
    model_folder = './model/'
    model_file = 'Key-LSTM-RAW'
    model_path = model_folder + model_file + '.pth'

    # データセットの生成
    test_EMG_dataset = dataset.EMGDatasetFFT(dataset_folder=dataset_folder,
                                             class_name=LABEL_NAMES,
                                             is_train=False)

    # dataloaderの生成
    Test_DataLoader = DataLoader(test_EMG_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 drop_last=True,
                                 pin_memory=True)

    # モデルの宣言と読み込み
    # net = model.EMG_Inference_Model_Linear(input_size=RAW_DATA_LENGTH).to(device)
    net = model.EMG_Inference_Model_LSTM(input_size=CH_NUM, hidden_size=int(RAW_DATA_LENGTH/4)).to(device)
    net.load_state_dict(torch.load(model_path))

    # ネットワークを評価モードにする
    net.eval()

    acc = 0.0
    all_predict_time = 0.0

    with tqdm(total=len(Test_DataLoader), unit='batch', desc='[test]')as pb:
        # 勾配の計算をしない
        with torch.no_grad():
            for data, label in Test_DataLoader:
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

    print('accuracy = {}'.format(acc/len(Test_DataLoader)))
    print('predict time = {}[sec]'.format(all_predict_time))


if __name__ == '__main__':
    main()
