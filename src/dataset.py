import os
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Any
from types import SimpleNamespace

import torch
from torch.utils.data.dataloader import Dataset
import torchvision.transforms as T

from .definitions import *


LABEL_MAP = {'No Finding': 0,
            'Enlarged Cardiomediastinum': 1,
            'Support Devices': 2,
            'Fracture': 3,
            'Lung Opacity': 4,
            'Cardiomegaly': 5,
            'Pleural Effusion': 6,
            'Pleural Other': 7,
            'Pneumothorax': 8,
            'Edema': 9,
            'Consolidation': 10,
            'Pneumonia': 11,
            'Lung Lesion': 12,
            'Atelectasis': 13}


class CheXpert(Dataset):
    def __init__(self, paths: Optional[str] = None, args: Optional[SimpleNamespace] = None, 
                 split: str = 'train', transform = None, trim=False):        
        self.split = split
        self.trim = trim
        
        # Labels for full and competetion models
        label_path: str = os.path.join(ROOT_DIR, 'info', 'labels.yaml')
        with open(label_path, "r") as file:
            labels_dict = yaml.safe_load(file)
        
        self.attr_all_names = labels_dict['std']
        self.attr_names = labels_dict['comp']
        
        # Setup transforms
        if transform is None:
            self.transform = self._initialize_transforms()
        else:
            self.transform = transform
            
        # Setup paths
        train_path: str = ""
        val_path: str = ""
        if args is not None:
            train_path = os.path.join(args.data_dir, "train.csv")
            val_path = os.path.join(args.data_dir, "valid_mod.csv")
        else:
            train_path, val_path = paths[0], paths[1]
            
        # Load and Setup the training and validation dataframes
        self.df = pd.DataFrame([None])
        if self.split.lower() == 'train':
            self.df = pd.read_csv(train_path)
        elif self.split.lower() in ['valid', 'val']:
            self.df = pd.read_csv(val_path)           
        else:
            self.df = pd.read_csv(train_path)
            temp_df = pd.read_csv(val_path)           
            self.df.append(temp_df)
        
        if trim:
            for attr in self.attr_all_names:
                if attr not in self.attr_names:
                    self.df = self.df.drop(attr, axis=1)
        
    def __len__(self):
        return len(list(self.df['Path']))
    
    def _initialize_transforms(self):
        MEAN = [0.485, 0.456, 0.406] 
        STD = [0.229, 0.224, 0.225]
        
        t = T.Compose([
            T.Resize((self.args.img_size, self.args.img_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.5, 0.5, 0.5, 0.5),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        
        return t
    
    def _get_label(self, index):
        if not self.trim:
            for c in list(self.df.columns):
                if self.df[c][index] == 1.0:
                    return  torch.tensor(LABEL_MAP[c], dtype=torch.long) 
        else:
            for c in list(self.df.columns):
                if self.df[c][index] == 1.0:
                    return torch.tensor(self.attr_names.index(c), dtype=torch.long)
    
    def __getitem__(self, index):
        path: str = self.df['Path'][index]
        label = self._get_label(index)
        image = Image.open(path).convert("RGB")
        
        if label is None:
            label = torch.tensor(0, dtype=torch.long)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class CheXpert2(Dataset):
    def __init__(self, data_dir: str, transforms: Any = None, split = "train", **kwargs):
        super().__init__()
        # Error Handling
        split_options: list = ['train', 'val', 'valid', 'test']
        if split.lower() not in split_options:
            raise ValueError("Invalid value for argument 'split',\n  'split' should be in {}".format(split_options))
        
        # Initialize paths and labels
        self._train_path: str = os.path.join(data_dir, "chexpert-train.csv")
        self._val_path: str = os.path.join(data_dir, "chexpert-valid.csv")
        self._test_path: str = os.path.join(data_dir, "chexpert-test.csv")
        self.till = kwargs.get("till", -1)
        self.policy = kwargs.get("policy", "ones")
        img_size = kwargs.get("img_size", 224)
        
        self._initialize_paths_and_labels(split)
                
        # Initialize the transforms
        if transforms is None:
            self._initialize_transforms(img_size)
        else:
            self.transforms = transforms
            
    def _initialize_transforms(self, img_size):
        MEAN = [0.485, 0.456, 0.406] 
        STD = [0.229, 0.224, 0.225]
        
        self.transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.5, 0.5, 0.5, 0.5),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    
    def __generate_replace_dicts(self):
        rp_dicts = []
        if self.policy == "ones":
            rp_dicts.append({l : -1 for l in self._LABELS})
            rp_dicts.append({l : 1 for l in self._LABELS})
        else:
            rp_dicts.append({l : -1 for l in self._LABELS})
            rp_dicts.append({l : 0 for l in self._LABELS})
        return rp_dicts
    
    def __load(self, path):
        # Load the metadata as dataframe
        self.df = pd.read_csv(path)[:self.till]
        
        # Load the datapaths to the images
        self._data_paths = self.df["Path"].to_numpy()
        self._data_paths = self._data_paths
        
        # Load the string-labels
        self._LABELS = []
        for idx, col in enumerate(self.df.columns):
            if idx > 4:
                self._LABELS.append(col)
        rp_dicts = self.__generate_replace_dicts()
        self.df = self.df.replace(*rp_dicts)
    
        # load the BCE LABELS
        self._labels = self.df[self._LABELS].to_numpy()
        self._labels = np.nan_to_num(self._labels)
        self._labels = self._labels

    def _initialize_paths_and_labels(self, split: str):
        if split.lower() == 'train':
            self.__load(self._train_path)
        elif split.lower() in ['val', 'valid']:
            self.__load(self._val_path)
        else:
            self.__load(self._test_path)
            
    def _get_labels(self, index):
        for idx, col in enumerate(self.df.columns):
            if idx > 4:
                if self.df[col][index] == 1 or self.df[col][index] == -1:
                    return LABEL_MAP[col]
            
    def __len__(self):
        return len(self._data_paths)
    
    def __getitem__(self, index):
        img = Image.open(self._data_paths[index]).convert('RGB')
        if self.policy in ['ones', 'zeros']:
            label = self._labels[index]
            label = torch.FloatTensor(label)
        else:
            label = self._get_labels(index)
            if label is not None:
                label = torch.tensor(label)
            else:
                label = torch.tensor(0)
            
        img = self.transforms(img)
        return img, label