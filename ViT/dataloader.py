import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class TECDataset:
    def __init__(self, path, mode = 'maxmin', patch_size = 4, target_hour = 24, input_history = 1, pretrained = False) -> None:
        self.path = path
        self.patch_size = patch_size
        self.target_hour = target_hour
        self.val, self.val2 = 0, 0
        self.input_history = input_history
        self.mode = "None"
        self.year = ""
        self.pretrained = pretrained
        self.tec_data = self.read_csv_data()
        if mode == 'maxmin':
            self.val, self.val2 = self.maxmin()
            self.mode = 'max_min'
        elif mode == 'z_score': #z_score
            self.val, self.val2 = self.z_scores()
            self.mode = 'z_score'
        self.DOY_info, self._input, self.target = [], [], []
        self.create_dataset()
    
    #Let the data go through normalization
    def maxmin(self) :
        max_tec = 500
        min_tec = 0
        self.tec_data = np.array(self.tec_data) / max_tec
        return max_tec, min_tec

    #Let the data go through standardization
    def z_scores(self) :
        standardizer  = StandardScaler()
        scaler = standardizer.fit(self.tec_data)
        self.tec_data = scaler.transform(self.tec_data)
        mean, std = scaler.mean_, scaler.scale_
        return std[0], mean[0]

    #read data from csv
    def read_csv_data(self):
        dataset = []
        for file_name in os.listdir(self.path):
            self.year += (file_name.split('.')[0] + ', ')
            list_col = [i for i in range(11, 5123)]
            list_col += [1, 2, 3]
            data = pd.read_csv(os.path.join(self.path, file_name), skiprows= 5, usecols = list_col)
            data = data.dropna(axis=0, how='any')
            data = data.values
            dataset.append(data)
        
        tec_data = dataset[0]
        for i in range(1, len(dataset)):
            tec_data = np.vstack((tec_data,dataset[i]))
        return tec_data

    #Create a dataset that fits the model
    def create_dataset(self) :
        for idx in tqdm(range(self.input_history-1, len(self.tec_data)), dynamic_ncols=True):
            if idx + self.target_hour>= len(self.tec_data):
                break
            input_history_TEC, target_TEC = [], []
            if not self._input:
                for i in range(self.input_history):
                    input_history_TEC.append(self.create_input(i))
            else:  
                input_history_TEC = self._input[-1][1:]
                input_history_TEC.append(self.create_input(idx))
            target_TEC.append(self.create_input(idx+self.target_hour))
            self._input.append(input_history_TEC)
            self.target.append(target_TEC)
            if self.pretrained:
                self.DOY_info.append(self.tec_data[idx][:3])
            else:
                self.DOY_info.append(self.tec_data[idx+self.target_hour][:3])
        return
    
    def create_input(self, idx):
        world = []
        for longitude in range(71):
            latitude = []
            for lat in range(72):
                latitude.append(self.tec_data[idx][3+longitude*72 + lat])
            world.append(latitude)
        world.append([0 for _ in range(72)])
        return world

    def create_target(self, idx):
        patch_count = 72*72//(self.patch_size*self.patch_size)
        target_world = [[]for _ in range(patch_count)]
        
        for longitude in range(71):
            for lat in range(72):
                patch_idx = (longitude//self.patch_size)*(72//self.patch_size) + lat//self.patch_size
                target_world[patch_idx].append(self.tec_data[idx + self.target_hour][3 + longitude*72 + lat])
        
        for patch_idx in range(patch_count - (72//self.patch_size), patch_count): #padding zero to the final latitude
            for _ in range(self.patch_size): 
                target_world[patch_idx].append(0)
        return target_world

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        DOY_info = torch.tensor(self.DOY_info[idx], dtype=torch.float)
        _input = torch.tensor(self._input[idx], dtype=torch.float)
        target = torch.tensor(self.target[idx], dtype=torch.float)
        return DOY_info, _input, target,


if __name__ == '__main__':
    dataset = TECDataset('../data/valid', mode = 'None', target_hour = 12, input_history = 1, patch_size=12)
    dataloader = DataLoader(dataset, batch_size=24)
    for data in dataloader:
        print(np.shape(data[0]))
        print(np.shape(data[1]))
        print(data)
        input()
        break