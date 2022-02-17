import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from zmq import device
from model import UNet
from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torchio as tio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import DiceLoss, check_accuracy

# hyperparameters
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False

def show_batch(dl):
    for images, labels in dl:
        print(len(images))
        print(images[0].shape)
        
        image = images[0].permute(1,2,0)
        label = labels[0]

        plt.imshow(label.squeeze(0), cmap="gray")
        
        plt.show()
        


def train_model(model,loaders,optimizer,num_of_epochs,loss_fn):
    
    # iterate through the epochs
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        train_losses = []

        # iterate through the batches
        for i, data in enumerate(loaders[0]):

            # put images into devices
            train_image, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)
            optimizer.zero_grad()

            # prediction
            out = model(train_image)

            # loss compared to actual
            loss = loss_fn(out,ground_truth)

            train_losses.append(loss)

            # backward prop and optimize
            loss.backward()
            optimizer.step()
            

            running_loss += loss.item()

        train_loss = torch.stack(train_losses).mean().item()
        epoch_loss = running_loss / 100
        print(f"Overall Epoch: {epoch} loss: {train_loss}")
        running_loss = 0.0

            


if __name__ == '__main__':

    #model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #in_channels=5, out_channels=1, init_features=32, pretrained=False)

    unet_2d = UNet() # make sure to change the number of channels in the unet model file
    print(DEVICE)
    unet_2d.to(DEVICE)


    # Need to consider splitting the training set manually
    train_directory = "ISLES/TRAINING"
    val_directory = "ISLES/VALIDATION"

    modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']

    # load our train and validation sets
    train_set = train_ISLES2018_loader(train_directory, modalities)
    print("Loaded Training Data")
    val_set = val_ISLES2018_loader(val_directory, modalities)
    print("Loaded Validation Data")


    #train_set, val_set = random_split(train_dataset, (400,102))

    print(len(train_set))
    print(len(val_set))

    # need to figure out how to import the data without issues with masks etc
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4 ,shuffle=True, pin_memory=True)
    valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4 ,shuffle=True, pin_memory=True)

    #print(train_dl['train'])

    optimizer = optim.Adam(unet_2d.parameters(), LEARNING_RATE)
    dsc_loss = DiceLoss()

    loss_fn = nn.BCEWithLogitsLoss()

    #show_batch(train_dl)
    train_model(unet_2d, (train_dl, valid_dl),optimizer,100,loss_fn)
