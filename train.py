import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from zmq import device
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
from utils import DiceLoss

# hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False


def train_model(model,loaders,optimizer,num_of_epochs,loss_fn):
    
    # iterate through the epochs
    for epoch in range(num_of_epochs):
        running_loss = 0.0

        # iterate through the batches
        for i, data in enumerate(loaders[0], 0):
            # put images into devices
            train_image, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()

            out = model(train_image)
            loss = loss_fn(out,ground_truth)

            # backward prop and optimize
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()

           
        print("loss for epoch " + str(epoch) + ":  " + str(running_loss))
            


if __name__ == '__main__':
    unet_2d = UNet() # make sure to change the number of channels in the unet model file
    print(DEVICE)
    unet_2d.to(DEVICE)

    train_directory = "ISLES/TRAINING"
    #test_directory = "ISLES/TESTING"

    modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
    train_dataset = ISLES2018_loader(train_directory, modalities)
    #test_dataset = ISLES2018_loader(test_directory, modalities)

    # split dataset into train/val for given modality
    #print(len(train_dataset))

    train_set, val_set = random_split(train_dataset, (400,102))
    #print(len(train_set[1]))
    #print(len(train_set),len(val_set))

    # need to figure out how to import the data without issues with masks etc
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4 ,shuffle=True, pin_memory=True)
    valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4 ,shuffle=True, pin_memory=True)

    #print(train_dl['train'])

    optimizer = optim.Adam(unet_2d.parameters(), LEARNING_RATE)
    dsc_loss = DiceLoss()

    loss_fn = nn.BCEWithLogitsLoss()

    train_model(unet_2d, (train_dl, valid_dl),optimizer,100,loss_fn)
