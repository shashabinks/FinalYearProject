import sklearn
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from zmq import device
from model import UNet
#from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
from case_dataloader import train_ISLES2018_loader,val_ISLES2018_loader, load_data
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torchio as tio
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from unet_model import UNet_2D
from utils import DiceLoss, check_accuracy, save_predictions_as_imgs, calc_loss, dc_loss

# hyperparameters
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 101
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False

total_train_loss = []

def accuracy():
    pass

def show_batch(dl):
    for images, labels in dl:
        print(len(images))
        print(images[0].shape)
        
        image = images[0].permute(1,2,0)
        label = labels[0]

        plt.imshow(label.squeeze(0), cmap="gray")
        
        plt.show()
        


def train_model(model,loaders,optimizer,num_of_epochs):
    
    # iterate through the epochs
    for epoch in range(num_of_epochs):
        running_loss = 0.0
        epoch_loss = 0.0
        train_losses = []
        train_accs = []

        # iterate through the batches
        for i, data in enumerate(loaders[0]):

            # put images into devices
            train_image, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)
            

            # prediction
            out = model(train_image)

            # loss compared to actual
            loss = calc_loss(out,ground_truth)

            # calculate dice coefficient
            out = torch.sigmoid(out)
            _,dice_coeff = dc_loss(out, ground_truth)
            

            train_losses.append(loss)
            train_accs.append(dice_coeff)

            # backward prop and optimize network
            loss.backward()
            optimizer.step()

            # reset gradients to zero in prep for next batch
            optimizer.zero_grad()

            #running_loss += loss.item()
        
        # test validation dataset after each epoch
        train_loss = torch.stack(train_losses).mean().item()
        train_acc = torch.stack(train_accs).mean().item()

        total_train_loss.append(train_loss)
        
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss} Train Acc: {train_acc} ")
        
        check_accuracy(loaders[1], model, device=DEVICE)

        # view images after 100 epochs
        if epoch % 25 == 0:
            save_predictions_as_imgs(loaders[1], model, folder="saved_images/", device=DEVICE)
        
        

            


if __name__ == "__main__":
    torch.cuda.empty_cache()

    unet_2d = UNet_2D() # make sure to change the number of channels in the unet model file
    print(DEVICE)

    # change this when u change model
    unet_2d.to(DEVICE)


    # Need to consider splitting the training set manually
    train_directory = "ISLES/TRAINING"
    val_directory = "ISLES/VALIDATION"

    modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']

    ### NEW STUFF ###
    directory = "ISLES/TRAINING"
    dataset = load_data(directory)

    train_data,val_data = train_test_split(dataset, test_size=0.2, train_size=0.8, shuffle=True)

    

    #################

    # load our train and validation sets
    train_set = train_ISLES2018_loader(train_data, modalities)
    print("Loaded Training Data")
    val_set = val_ISLES2018_loader(val_data, modalities)
    print("Loaded Validation Data")

    print(len(train_set))
    print(len(val_set))

    # need to figure out how to import the data without issues with masks etc
    train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=True, pin_memory=True)
    valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=False, pin_memory=True)

    #print(train_dl['train'])

    optimizer = optim.Adam(unet_2d.parameters(), LEARNING_RATE)

    train_model(unet_2d, (train_dl, valid_dl),optimizer,NUM_EPOCHS)

    plt.plot(total_train_loss)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Training Loss")
    plt.show()


