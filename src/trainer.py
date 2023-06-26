import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(path)

import time
import random
from tqdm.auto import tqdm
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Optional, Union, Any, List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchmetrics
import torchvision
import torchvision.transforms as T

from hxe.losses import HierarchicalXELoss
from src.dataset import CheXpert
from src.cx_models import load_base_classifier


class Trainer:
    def  __init__(self, args: SimpleNamespace, custom_transforms: List[list] = None):
        self.args = args
        
        if custom_transforms is None:
            self.__setup_transforms()
        else:
            self.train_transforms = T.Compose(custom_transforms[0])
            self.val_transforms =  T.Compose(custom_transforms[1])

        self._warmup()
        self.history = []
        
    def _warmup(self):
        print("Warming up the Trainer !")
        self.device = torch.device('cuda') if self.args.use_cuda else torch.device('cpu')
        self.__setup_dirs()
        self.__setup_data()
        print("Loading Model ...")
        self.network = load_base_classifier(self.args.model_name, self.args)
        self.network = self.network.to(self.device)
        self.__setup_aux_params()
        
    def __setup_dirs(self):
        print("Setting up Directories ...")
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        
        self.save_path = os.path.join(self.args.save_dir, self.args.model_name + "_hxe.pth")
        
    def __setup_data(self):
        print("Preparing data ...")
        train_path: str = os.path.join(self.args.data_dir, "train.csv")
        val_path: str = os.path.join(self.args.data_dir, "valid_mod.csv")
        
        self.train_ds = CheXpert(paths = [train_path, val_path], split="train", transform=self.train_transforms)
        self.val_ds = CheXpert(paths = [train_path, val_path], split="val", transform=self.val_transforms)
        
        self.train_dl = DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, 
                                   pin_memory=self.args.pin_memory, num_workers=self.args.num_workers)
        self.val_dl = DataLoader(self.val_ds, batch_size=self.args.batch_size, shuffle=True, 
                                   pin_memory=self.args.pin_memory, num_workers=self.args.num_workers)
    
    def __setup_aux_params(self):
        print("Setting up optimizer ...")
        self.optimizer = None
        
        # Setup the Optimizer
        if self.args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args.lr, betas=self.args.betas)
        elif self.args.opt == 'sgd':
            self.optimizer == torch.optim.SGD(self.network.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            raise NotImplementedError("Optimizer not present / invalid")

        # Setup the Loss function
        d = 'gpu' if self.args.use_cuda else 'cpu'
        self.criterion = HierarchicalXELoss(device=d)

    def __setup_transforms(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
        IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)
        img_size = self.args.img_size

        transforms_list = []
        transforms_list.append(T.Resize((img_size, img_size)))
        transforms_list.append(T.RandomResizedCrop((img_size, img_size)))
        transforms_list.append(T.RandomHorizontalFlip())
        transforms_list.append(T.ToTensor())
        transforms_list.append(T.Normalize(IMAGENET_MEAN, IMAGENET_STD))

        self.train_transforms = T.Compose(transforms_list)
        self.val_transforms = T.Compose(transforms_list)
        
    def _save(self, state):
        torch.save(state, self.save_path)

    def _summary(self, epoch, train_loss, val_loss, auroc_indv, auroc_mean):
        print(f"Epoch: {epoch+1}/{self.args.epochs}:")
        print(f"\tTrain-Loss: {train_loss} || Val-Loss: {val_loss}")
        print(f"\tAUROC-Individual: {auroc_indv}")
        print(f"\tAUROC-Mean: {auroc_mean}")

    def _train_one_epoch(self):
        train_loss = 0
        num_training_steps = self.args.epochs*len(self.train_dl)
        progress_bar = tqdm(range(num_training_steps))
        self.network.train()
        for bid, (images, labels) in tqdm(enumerate(self.train_dl)):
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits = self.network(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            progress_bar.update(1)
        
        return train_loss / len(self.train_dl)
        
    def _validate(self):
        eval_loss = 0
        gts = torch.FloatTensor().cuda()
        preds = torch.FloatTensor().cuda()
        num_val_steps = self.args.epochs*len(self.train_dl) 
        progress_bar = tqdm(range(num_val_steps))

        self.network.eval()
        with torch.no_grad():
            for bid, (images, labels) in tqdm(enumerate(self.val_dl)):
                labels = labels.to(self.device)
                images = images.to(self.device)
                gts = torch.cat((gts, labels), 0)

                logits = self.network(images)

                preds = torch.cat((preds, logits), 0)
                eval_loss = self.criterion(logits, labels)
            
            auroc_indv = self.compute_metrics(gts, preds)
            auroc_mean = np.array(auroc_indv).mean()
            progress_bar.update(1)            
        
        return eval_loss / len(self.val_dl), auroc_indv, auroc_mean
    
    def plot(self):
        # Prepare the data
        ## epochs
        epochs = list(range(self.args.epochs))
        ## Losses
        train_loss = [result['train_loss'].cpu().item() for result in self.history]
        val_loss = [result['val_loss'].cpu().item() for result in self.history]
        
        ## Individual ROCs
        col_ind_rocs = [result['auroc_indv'] for result in self.history]
        col_ind_rocs = np.array(col_ind_rocs)
        col_ind_rocs = col_ind_rocs.reshape((14, -1))
        mean_ind_rocs = np.sum(col_ind_rocs) / col_ind_rocs.shape[0]
        ## Mean ROCs
        mean_rocs = [result['auroc_mean'] for result in self.history]
        
        # PLOT
        fig = plt.figure(figsize=(9, 16))
        ## Loss Subplots
        ax1 = fig.add_subplot(131)
        ax1.plot(epochs, train_loss, color='blue')
        ax1.plot(epochs, val_loss, color='red')
        ax1.legend(['TrainLoss, ValLoss'])
        ax1.set(xlabel='epochs', ylabel='loss')
        ax1.set_title('Loss vs Epochs')
        
        ## ROC Subplots
        ### Individual ROCs
        ax2 = fig.add_subplot(132)
        ax2.bar(self.args.labels, mean_ind_rocs)
        ax2.set_title('Individual ROCs')
        ### Mean Rocs
        ax3 = fig.add_subplot(133)
        ax3.plot(epochs, mean_rocs, color='orange')
        ax3.set(xlabel='epochs', ylabel='mean-roc')
        ax3.set_title('Mean ROC vs Epochs')
        
        sp = os.path.join(self.args.save_dir, 'plot.png')
        plt.savefig(sp)

    def compute_metrics(self, ground_truths, preds):
        AUROC = []

        gt_oh = F.one_hot(ground_truths.cpu().to(torch.long), self.args.num_classes)
        pred_cpu = preds.cpu()
        # print("gt: {} | pred: {}".format(gt_oh.size(), pred_cpu.size()))
        # print(gt_oh)
        
        auroc = torchmetrics.AUROC(self.args.num_classes)
        for i in range(self.args.num_classes):
            try:
                AUROC.append(auroc(pred_cpu[i, ], gt_oh[i, ]))
            except ValueError:
                pass

        return AUROC

    def train(self):
        print("Training the Network")
        best_auc = 0
        for epoch in range(self.args.epochs):
            train_loss = self._train_one_epoch()
            val_loss, auroc_indv, auroc_mean = self._validate()

            if auroc_mean > best_auc:
                best_auc = auroc_mean
                state = {
                    'epoch': epoch + 1,
                    'state_dict': self.network.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_auc': best_auc,
                }
                self._save(state)

            self._summary(epoch, train_loss, val_loss, auroc_indv, auroc_mean)
            result = {'epoch': epoch,
                      'train_loss': train_loss,
                      'val_loss': val_loss,
                      'auroc_indv': auroc_indv, 
                      'auroc_mean': auroc_mean,
                      'best_auc': best_auc}
            self.history.append(result)
            
        return self.history