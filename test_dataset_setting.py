import os
from settings import *

#テストデータは連番の210-229
def test_generate():
    for i in range(210):
        str_i = str(i).zfill(4)
        file_name = str_i + '.csv'
        dataset_folder = os.path.join('dataset', 'test')
        
        for c in range(8):
            ch_num = 'ch' + str(c)
            file_path_index = os.path.join(dataset_folder, 'index', ch_num, file_name)
            file_path_middle = os.path.join(dataset_folder, 'middle', ch_num, file_name)
            file_path_pinkie = os.path.join(dataset_folder, 'pinkie', ch_num, file_name)
            file_path_ring = os.path.join(dataset_folder, 'ring', ch_num, file_name)
            file_path_thumb = os.path.join(dataset_folder, 'thumb', ch_num, file_name)
            os.remove(file_path_index)
            os.remove(file_path_middle)
            os.remove(file_path_pinkie)
            os.remove(file_path_ring)
            os.remove(file_path_thumb)


def test_rename():
    for finger in LABEL_NAMES:
        for i in range(20):
            rename_str_i = str(i).zfill(4)
            now_str_i = str(i+210).zfill(4)
            rename_file_name = rename_str_i + '.csv'
            now_file_name = now_str_i + '.csv'
            dataset_folder = os.path.join('dataset', 'test')
            for c in range(8):
                ch_num = 'ch' + str(c)
                now_file_path = os.path.join(dataset_folder, finger, ch_num, now_file_name)
                rename_file_path = os.path.join(dataset_folder, finger, ch_num, rename_file_name)
                os.rename(now_file_path, rename_file_path)


if __name__ == '__main__':
    test_rename()
