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
        img_clip_path = self.df.loc[index]['path']
        ev_clip_path = img_clip_path.replace('rgb', 'event_thr_10')
        img_clip_feature = np.load(img_clip_path)
        ev_clip_feature = np.load(ev_clip_path)

        # test loader
        if self.test_mode == False:
            img_clip_feature, img_clip_length = process_feat(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_feat(ev_clip_feature, self.clip_dim)
        # train_loader
        else:
            img_clip_feature, img_clip_length = process_split(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_split(ev_clip_feature, self.clip_dim)

        img_clip_feature = torch.tensor(img_clip_feature)
        ev_clip_feature = torch.tensor(ev_clip_feature)
        clip_label = self.df.loc[index]['label']
        return img_clip_feature, ev_clip_feature, clip_label, img_clip_length


class XD_Dataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_clip_path = self.df.loc[index]['path']
        ev_clip_path = img_clip_path.replace('rgb', 'event_thr_10')
        img_clip_feature = np.load(img_clip_path)
        ev_clip_feature = np.load(ev_clip_path)
        
        if self.test_mode == False:
            img_clip_feature, img_clip_length = process_feat(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_feat(ev_clip_feature, self.clip_dim)
        else:
            img_clip_feature, img_clip_length = process_split(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_split(ev_clip_feature, self.clip_dim)
        
        img_clip_feature = torch.tensor(img_clip_feature)
        ev_clip_feature = torch.tensor(ev_clip_feature)
        clip_label = self.df.loc[index]['label']
        return img_clip_feature, ev_clip_feature, clip_label, img_clip_length


class Shang_Dataset(data.Dataset):
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
            self.df = self.df.loc[self.df['label'] == 'normal']
            self.df = self.df.reset_index()
        # train abnormal
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'normal']
            self.df = self.df.reset_index()
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_clip_path = self.df.loc[index]['path']
        ev_clip_path = img_clip_path.replace('rgb', 'event')
        img_clip_feature = np.load(img_clip_path)
        ev_clip_feature = np.load(ev_clip_path)

        # test loader
        if self.test_mode == False:
            img_clip_feature, img_clip_length = process_feat(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_feat(ev_clip_feature, self.clip_dim)
        # train_loader
        else:
            img_clip_feature, img_clip_length = process_split(img_clip_feature, self.clip_dim)
            ev_clip_feature, ev_clip_length = process_split(ev_clip_feature, self.clip_dim)

        img_clip_feature = torch.tensor(img_clip_feature)
        ev_clip_feature = torch.tensor(ev_clip_feature)
        clip_label = self.df.loc[index]['label']
        return img_clip_feature, ev_clip_feature, clip_label, img_clip_length