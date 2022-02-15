import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from model import UNet
from datasetloader import ISLES2018_loader
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torchio as tio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  #      this can be changed to fit ur data
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False

unet_3d = UNet()
unet_3d.to(DEVICE)

directory = "ISLES/TRAINING"
modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset = ISLES2018_loader(directory, modalities)

# split dataset into train/val for given modality
print(len(dataset))

train_set, val_set = random_split(dataset, (400,102))
print(len(train_set[1]))
print(len(val_set))

# need to figure out how to import the data without issues with masks etc
train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=2 ,shuffle=False, pin_memory=True)
valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=2 ,shuffle=False, pin_memory=True)

#print(tr)

optimizer = optim.Adam(unet_3d.parameters(), LEARNING_RATE)
