import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from utils.tools import StandardScaler

import warnings
warnings.filterwarnings('ignore')


class Dataset_TX(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='case2000.npz'):

        # info
        self.seq_len = size[0]

        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # df_raw = pd.read_csv(os.path.join(self.root_path,
        #                                   self.data_path))
        
        raw_data = np.load(os.path.join(self.root_path,
                                          self.data_path))
        
        inputs = torch.tensor(raw_data['z_att_noise'], dtype=torch.float32)
        labels = torch.tensor(raw_data['posi_label'], dtype=torch.float32)

        len_inputs = int(inputs.shape[0])

        num_train = int(len_inputs * 0.7)
        num_test = int(len_inputs * 0.2)
        num_vali = len_inputs - num_train - num_test
        border1s = [0, num_train - self.seq_len, len_inputs - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len_inputs]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = inputs[border1:border2]
        self.data_y = labels[border1:border2]
        # self.data_time = input_time[border1:border2]

        self.day_time = day_steps_tensor[border1:border2]
        self.week_time = week_steps_tensor[border1:border2]


    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, index):

        seq_x = self.data_x[index]
        label_y = self.data_y[index]

        return seq_x, label_y
    