import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DATASegLoader(Dataset):
    def __init__(self, args, root_path, data_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(root_path + data_path, dtype={'cols': str})
        data['cols'] = data['cols'].apply(lambda x: 'label' if 'label' in str(x).lower() else x)
        min_len = data['cols'].value_counts().min()
        data_trimmed = data.groupby('cols').head(min_len).copy()
        data_trimmed['row'] = data_trimmed.groupby('cols').cumcount()
        data = data_trimmed.pivot(index='row', columns='cols', values='data')
        if 'label' in data.columns:
            label_col = data.pop('label')
            data['label'] = label_col
        data = data.reset_index(drop=True)
        data = data.to_numpy()

        train_len = get_series_info(args.data_path, 'train_lens')
        train, test, self.test_labels = data[:train_len, :-1], data[train_len:, :-1], data[train_len:, -1]

        self.scaler.fit(train)
        self.train = self.scaler.transform(train)
        self.test = self.scaler.transform(test)
        self.val = self.train[(int)(train_len * 0.8):]

        if args.c_in == 1:
            self.train = self.train[:, 0:1]
            self.val = self.val[:, 0:1]
            self.test = self.test[:, 0:1]
        print("train:", self.train.shape)
        print("val:", self.val.shape)
        print("test:", self.test.shape)
        print("test_labels:", self.test_labels.shape)

    def __len__(self):
        if self.flag == "train":
            len = (self.train.shape[0] - 2 * self.win_size) // self.step + 1
            return len
        elif (self.flag == 'val'):
            len = (self.val.shape[0] - 2 * self.win_size) // self.step + 1
            return len
        elif (self.flag == 'test'):
            len = (self.test.shape[0] - 2 * self.win_size) // self.step + 1
            return len
        else:
            len = (self.test.shape[0] - 2 * self.win_size) // self.step + 1
            return len

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return (np.float32(self.train[index:index + self.win_size]),
                    np.float32(self.train[index + self.win_size:index + 2 * self.win_size]))
        elif (self.flag == 'val'):
            return (np.float32(self.val[index:index + self.win_size]),
                    np.float32(self.val[index + self.win_size:index + 2 * self.win_size]))
        elif (self.flag == 'test'):
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test[index + self.win_size: index + 2 * self.win_size]),
                    np.float32(self.test_labels[index + self.win_size: index + 2 * self.win_size]))
        else:
            return (np.float32(self.test[index:index + self.win_size]),
                    np.float32(self.test[index + self.win_size: index + 2 * self.win_size]),
                    np.float32(self.test_labels[index + self.win_size:index + 2 * self.win_size]))


def get_series_info(data:str, info_field:str):
    series_info = pd.read_csv('./dataset/DETECT_META.csv')
    row = series_info[series_info['file_name'] == data]
    if not row.empty and info_field in series_info.columns:
        return row.iloc[0][info_field]
    else:
        return None
