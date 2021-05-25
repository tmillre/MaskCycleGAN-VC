"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import numpy as np
import os.path
import sys
import librosa
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import IPython

import soundfile

np.random.seed(111)

""""""
class DataSet:
    """
    Args:
        fold : which dataset to load: train A, train B, test A, test B
        sampleLen : how long of an audio sample to return in melgan windows (64 is default)
        eps : numerical factor to prevent division by 0 (default to 1e-10)

    """
    

    def __init__(self, fold="A", sampleLen = 64,
                 eps = 1e-10, aFolder = None, bFolder = None, atestFolder = None, btestFolder = None):
        
        if aFolder == None:
            aFolder = "vcc2018_database_training/vcc2018_training/VCC2SF1"
        if bFolder == None:
            bFolder = "vcc2018_database_training/vcc2018_training/VCC2SM2"
        if atestFolder == None:
            atestFolder = "vcc2018_database_evaluation/vcc2018_evaluation/VCC2SF1"
        if btestFolder == None:
            btestFolder = "vcc2018_database_evaluation/vcc2018_evaluation/VCC2SM2"
        
        fold = fold.upper()
        
        self.sL = sampleLen

        self.A = False
        self.B = False
        self.Test = False
        
        print(fold)

        if fold == "A":
            self.A = True
        elif fold == "B":
            self.B = True
        elif fold == "ATEST":
            self.A = True
            self.Test = True
        elif fold == "BTEST":
            self.B = True
            self.Test = True
        else:
            raise RuntimeError("Not train-val-test")

        #self.transform = transform
        #self.target_transform = target_transform
        
        melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

        # now load the corresponding dataset
        self.train_data = []
        if self.A and self.Test == False:
            for audio in os.listdir(aFolder):
                sample, fs = librosa.load(aFolder + '/' + audio)
                self.train_data.append(melgan(torch.tensor(np.expand_dims(sample, axis = 0))).cpu())
        elif self.B and self.Test == False:
            for audio in os.listdir(bFolder):
                sample, fs = librosa.load(bFolder + '/' + audio)
                self.train_data.append(melgan(torch.tensor(np.expand_dims(sample, axis = 0))).cpu())
        elif self.A and self.Test:
            for audio in os.listdir(atestFolder):
                sample, fs = librosa.load(atestFolder + '/' + audio)
                self.train_data.append(melgan(torch.tensor(np.expand_dims(sample, axis = 0))).cpu())
        elif self.B and self.Test:
            for audio in os.listdir(btestFolder):
                sample, fs = librosa.load(btestFolder + '/' + audio)
                self.train_data.append(melgan(torch.tensor(np.expand_dims(sample, axis = 0))).cpu())

        self.mu = np.mean(np.concatenate(self.train_data, axis=2), axis=2, keepdims=True)[0,:,:]
        self.std = np.std(np.concatenate(self.train_data, axis=2), axis=2, keepdims=True)[0,:,:] + eps
        
        print(self.mu.shape, self.std.shape)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (mfcc, mu, std) where mu and std specify the mean and standard deviation of the dataset.
        """
        #if training dataset, return random sample of length 64
        if not self.Test:
          hold = self.train_data[index][0,:,:]
        
          st_dex = 0
        
          if hold.shape[1] > self.sL:
            st_dex = np.random.randint(0, hold.shape[1] - self.sL)
        
          base_tensor = torch.Tensor(hold[: , st_dex:(st_dex + self.sL)])
        
          base_tensor = (base_tensor - self.mu)/ self.std
        
          return torch.unsqueeze(base_tensor, 0), self.mu, self.std

        #else, return entire sample
        else:
          hold = self.train_data[index][0,:,:]
        
          base_tensor = torch.Tensor(hold)
        
          base_tensor = (base_tensor - self.mu)/ self.std
        
          return torch.unsqueeze(base_tensor, 0), self.mu, self.std

    def __len__(self):
        return len(self.train_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
