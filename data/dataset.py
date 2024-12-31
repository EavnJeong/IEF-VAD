import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
from .tools import process_feat, process_split


class UCF_Dataset(data.Dataset):
    def __init__(
            self, 
            clip_dim: int, 
            file_path: str, 
            test_mode: bool, 
            label_map: dict, 
            normal: bool = False
        ):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        # train normal
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']
            self.df = self.df.reset_index()
        # train abnormal
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        # test loader
        if self.test_mode == False:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_dim)
        # train_loader
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


class XD_Dataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length