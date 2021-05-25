"""Headers"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

import matplotlib
import matplotlib.pyplot as plt
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

melgan = torch.hub.load('descriptinc/melgan-neurips', 'load_melgan')

def showAudio(mel, mean, std, filename = None):
    y = melgan.inverse((mel[0,0, :, :] * std[0,:,:] + mean[0,:,:]).unsqueeze(0)).cpu()
    if filename != None:
        soundfile.write(filename, y, 44100 // 2)
    IPython.display.display(IPython.display.Audio(y , rate = 44100 / 2))
