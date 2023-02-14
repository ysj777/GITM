import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class TECDataset:
    def __init__(self, path, mode = 'maxmin', window_size = 24) -> None:
        self.path = path
        self.window_size = window_size
        self.val, self.val2 = 0, 0
        self.mode = "None"
        self.tec_data = self.get_data()
        if mode == 'maxmin':
            self.val, self.val2 = self.maxmin()
            self.mode = 'max_min'
        elif mode == 'z_score': #z_score
            self.val, self.val2 = self.z_scores()
            self.mode = 'z_score'
        self._input, self.target = self.create_dataset()
    
    #Let the data go through normalization
    def maxmin(self) :
        ma = max(self.tec_data)
        mi = min(self.tec_data)
        min_max_scaler = MinMaxScaler()
        self.tec_data = min_max_scaler.fit_transform(self.tec_data)
        return ma[0], mi[0]

    #Let the data go through standardization
    def z_scores(self) :
        standardizer  = StandardScaler()
        scaler = standardizer.fit(self.tec_data)
        self.tec_data = scaler.transform(self.tec_data)
        mean, std = scaler.mean_, scaler.scale_
        return std[0], mean[0]

    #read data from csv
    def get_data(self):
        dataset = []
        for file_name in os.listdir(self.path):
            list_col = np.arange(5123)
            data = pd.read_csv(os.path.join(self.path, file_name), skiprows= 5, usecols = list_col)
            data = data.values
            dataset.append(data[:, 11:])
        
        tec_data = dataset[0]
        for i in range(1, len(dataset)):
            tec_data = np.vstack((tec_data,dataset[i]))
        return tec_data

    #Create a dataset that fits the model
    def create_dataset(self) :
        _input, target = [], []
        for idx in range(len(self.tec_data)):
            if idx + self.window_size>= len(self.tec_data):
                break
            word = []
            target_word = []
            for longitude in range(71):
                latitude = []
                target_latitude = []
                for lat in range(72):
                    latitude.append(self.tec_data[idx][longitude*72 + lat])
                    target_latitude.append(self.tec_data[idx + self.window_size][longitude*72 + lat])
                word.append(latitude)
                target_word.append(target_latitude)
            word.append([0 for _ in range(72)])
            target_word.append([0 for _ in range(72)])
            _input.append([word])
            target.append([target_word])
        return np.array(_input), np.array(target)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        _input = torch.tensor(self._input[idx], dtype=torch.float)
        target = torch.tensor(self.target[idx], dtype=torch.float)
        return _input, target,


if __name__ == '__main__':
    dataset = TECDataset('../data/train', mode = 'None', window_size = 24)
    dataloader = DataLoader(dataset, batch_size=24)
    for data in dataloader:
        print(np.shape(data[0]))
        print(data)
        break