import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset

from settings import *

#データセットの定義
class EMGDataset(Dataset):
    def __init__(self, dataset_folder='./dataset',
                 class_name='hoge',
                 is_train=True):
        self.dataset_folder = dataset_folder
        self.class_name = class_name
        self.is_train = is_train
        self.data_path, self.correct_class = self._get_file_names()

    def __getitem__(self, idx):
        emg_data_path, correct_class = self.data_path[idx], self.correct_class[idx]
        emg_sensors_data = []

        #データの読み込み
        for i in range(len(emg_data_path)):
            emg_csv_data = []
            with open(emg_data_path[i]) as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    emg_csv_data.append(float(row[0]))
            emg_data = np.array(emg_csv_data)

            #FFTの実行、負の周波数の分もそのまま使う(使わない場合はN/2する)
            emg_fft_data = np.abs(np.fft.fft(emg_data)).astype('float32')
            max_emg_val = np.max(emg_data)
            min_emg_val = np.min(emg_data)

            #振幅の正規化
            for i in range(len(emg_fft_data)):
                emg_fft_data[i] = (emg_data[i] - min_emg_val) / (max_emg_val - min_emg_val)
            
            emg_sensors_data.append(emg_fft_data)
        return emg_sensors_data, correct_class

    def __len__(self):
        #データ数を返す
        return len(self.data_path)

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        data_num_per_label = {}
        emg_data_path, correct_class = [], []
        one_label_path = []
        data_num = 0

        #データ数を記録
        for cname in self.class_name:
            data_path = os.path.join(self.dataset_folder, phase, cname, 'ch0')
            data_num_per_label[cname] = self.checkNumberOfFiles(data_path)

        #一つの指に対して8chあるため、ひとかたまりにする
        for cname in self.class_name:
            for data_num in range(data_num_per_label[cname]):
                data_name = str(data_num).zfill(4) + '.csv'

                for ch in range(8):
                    #csvデータが存在するか確認してパスをリストへ追加する
                    emg_data_dir = os.path.join(self.dataset_folder, phase, cname, 'ch'+str(ch))
                    data_name_path = os.path.join(emg_data_dir, data_name)
                    if not os.path.exists(data_name_path):
                        break
                    one_label_path.append(data_name_path)
                
                data_num += 1

                if len(one_label_path) != 0:
                    emg_data_path.append(one_label_path)
                    correct_class.append(LABEL_NAMES_DICT[cname])
                
                one_label_path = []

        assert len(emg_data_path) == len(correct_class), 'number of emg_data and class are not same.'
        #print(list(emg_data_path))
        return list(emg_data_path), list(correct_class)
    
    def checkNumberOfFiles(self, folder_path):
        file_count = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            if os.path.isfile(file_path):
                file_count +=1
        return file_count


#データセットの定義
class EMGDatasetLSTM(Dataset):
    def __init__(self, dataset_folder='./dataset',
                 class_name='hoge',
                 is_train=True):
        self.dataset_folder = dataset_folder
        self.class_name = class_name
        self.is_train = is_train
        self.data_path, self.correct_class = self._get_file_names()

    def __getitem__(self, idx):
        emg_data_path, correct_class = self.data_path[idx], self.correct_class[idx]
        emg_sensors_data = []

        #データの読み込み
        for i in range(len(emg_data_path)):
            emg_csv_data = []
            with open(emg_data_path[i]) as f:
                csv_reader = csv.reader(f)
                for row in csv_reader:
                    emg_csv_data.append(float(row[0]))
            emg_data = np.array(emg_csv_data)

            #FFTの実行、負の周波数の分もそのまま使う(使わない場合はN/2する)
            emg_fft_data = np.abs(np.fft.fft(emg_data)).astype('float32')
            max_emg_val = np.max(emg_data)
            min_emg_val = np.min(emg_data)

            #振幅の正規化
            for i in range(len(emg_fft_data)):
                emg_fft_data[i] = (emg_data[i] - min_emg_val) / (max_emg_val - min_emg_val)
            
            emg_sensors_data.append(emg_fft_data)
        
        emg_sensors_data_tensor = torch.tensor(emg_sensors_data)
        emg_sensors_data_tensor = torch.t(emg_sensors_data_tensor)

        return emg_sensors_data_tensor, correct_class

    def __len__(self):
        #データ数を返す
        return len(self.data_path)

    def _get_file_names(self):
        phase = 'train' if self.is_train else 'test'
        data_num_per_label = {}
        emg_data_path, correct_class = [], []
        one_label_path = []
        data_num = 0

        #データ数を記録
        for cname in self.class_name:
            data_path = os.path.join(self.dataset_folder, phase, cname, 'ch0')
            data_num_per_label[cname] = self.checkNumberOfFiles(data_path)

        #一つの指に対して8chあるため、ひとかたまりにする
        for cname in self.class_name:
            for data_num in range(data_num_per_label[cname]):
                data_name = str(data_num).zfill(4) + '.csv'

                for ch in range(8):
                    #csvデータが存在するか確認してパスをリストへ追加する
                    emg_data_dir = os.path.join(self.dataset_folder, phase, cname, 'ch'+str(ch))
                    data_name_path = os.path.join(emg_data_dir, data_name)
                    if not os.path.exists(data_name_path):
                        break
                    one_label_path.append(data_name_path)
                
                data_num += 1

                if len(one_label_path) != 0:
                    emg_data_path.append(one_label_path)
                    correct_class.append(LABEL_NAMES_DICT[cname])
                
                one_label_path = []

        assert len(emg_data_path) == len(correct_class), 'number of emg_data and class are not same.'
        #print(list(emg_data_path))
        return list(emg_data_path), list(correct_class)
    
    def checkNumberOfFiles(self, folder_path):
        file_count = 0
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)
            if os.path.isfile(file_path):
                file_count +=1
        return file_count