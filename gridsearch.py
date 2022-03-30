from collections import defaultdict
from math import gamma
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable


from patient_dataloader import train_ISLES2018_loader,val_ISLES2018_loader, load_data


import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, KFold
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR
import os



# MODELS
from models.RPDnet import RPDNet

h_params = {"lr":[0.00001,0.0001, 0.001, 0.01], "batch_size":[2,4], "epochs":[100]}


# hyperparameters
LEARNING_RATE = 0.0001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 200+1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
TRANSFORMER = False
DEEP_SUPERVISION = False


metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], 
            "train_precision":[],"val_precision":[], "train_recall":[], "val_recall":[], "train_hd":[], "val_hd":[]}


def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
    if hasattr(layer, 'reset_parameters'):
        #print(f'Reset trainable parameters of layer = {layer}')
        layer.reset_parameters()
      
# define training function
def train_model(model,loaders,optimizer,num_of_epochs,scheduler=None):

    #loss_fn = SymmetricUnifiedFocalLoss()
    best_score = -1.0

    loss = None
    
    # iterate through the epochs
    for epoch in range(0,num_of_epochs):

        epoch_samples = 0
        curr_metrics = defaultdict(float)

        # iterate through the batches
        for i, data in enumerate(loaders[0]):
            
            
            # put images into devices
            train_image, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

            # reset gradients to zero in prep for next batch
            optimizer.zero_grad()
    
            epoch_samples += train_image.size(0) # batch num
            
            
            # prediction
            out = model(train_image)

            if DEEP_SUPERVISION:
                loss = multi_loss_function(out,ground_truth,curr_metrics)

            else:
                # loss compared to actual
                loss = calc_loss(out,ground_truth, curr_metrics)

            # backward prop and optimize network
            loss.backward()
            optimizer.step()

        
        # test validation dataset after each epoch
        train_loss = curr_metrics['loss'] / epoch_samples
        train_acc = curr_metrics["dice_coeff"] / epoch_samples
        train_bce = curr_metrics['bce'] / epoch_samples
        
       

        metrics["train_loss"].append(train_loss)
        metrics["train_dice"].append(train_acc)
        metrics["train_bce"].append(train_bce)
        
        
        
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss} Train Dice Score: {train_acc} Train BCE: {train_bce}")
        
        epoch_val_acc = 0.0


        # check validation
        if DEEP_SUPERVISION:
            epoch_val_acc = multi_check_accuracy(loaders[1], model, device=DEVICE)

        else:
            epoch_val_acc = check_accuracy(loaders[1], model, device=DEVICE)

        if epoch_val_acc > best_score:
            print("Best Accuracy so far!, Saving model...")
            best_score = epoch_val_acc
            torch.save(model.state_dict(), os.path.join("Weights/", "best_weights.pth"))


        
        # view images after all epochs
        if epoch % (NUM_EPOCHS-1) == 0:
            if epoch == NUM_EPOCHS - 1:
                print("Final Epoch!, Saving model...")

                torch.save(model.state_dict(), os.path.join("Weights/", "final_weights.pth"))
        
        # check for decay
        if scheduler:
            scheduler.step()
    
    del loss


def view_images_multi(loader, model, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)[0])
        
        if idx == 1 or idx == 2 or idx == 4 or idx == 6 or idx == 8:
            f, (ax2, ax3) = plt.subplots(1, 2, figsize=(10,20))
            
            ax2.imshow(preds[0].cpu().squeeze().numpy(), cmap = 'gray',label="Prediction")
            ax3.imshow(y[0].squeeze(0), cmap= 'gray', label="Mask")
            #plt.legend()
            plt.show()

    model.train()
        
        
# calculate dice coefficient/loss
def dc_loss(inputs,targets,smooth=1.):
    inputs = inputs.contiguous()

    targets = targets.contiguous()
    
    intersection = (inputs * targets).sum(dim=2).sum(dim=2)                              
    dice = (2.*intersection + smooth)/(inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)  

    loss = 1 - dice
    
    return loss.mean(),dice.mean() # output loss

def to_one_hot(tensor,nClasses):
  n,c,h,w = tensor.size()
  one_hot = torch.zeros(n,nClasses,h,w).cuda().scatter_(1,tensor.view(n,1,h,w),1)
  return one_hot



# weighted due to class imbalance
def calc_bce(pred=None, target=None):
    bceweight = torch.ones_like(target)  +  20 * target # create a weight for the bce that correlates to the size of the lesion
    bce = F.binary_cross_entropy_with_logits(pred,target, weight=bceweight) # the size of the lesions are small therefore it is important to use this
    
    return bce


def focal_loss(pred, targets,alpha,gamma):
    
    bceweight = torch.ones_like(targets)  +  20 * targets # create a weight for the bce that correlates to the size of the lesion
    
    
    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy_with_logits(pred, targets,weight=bceweight)
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
    return focal_loss
    


# separate this bit and move the dc loss function into the train.py file...
# calculate weighted loss
def calc_loss(pred, target, curr_metrics):

    bce_weight = 0.5
    
    bce = calc_bce(pred,target)
    
    pred = torch.sigmoid(pred)

    
    dice,dice_coeff = dc_loss(pred, target)

    # combo loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    curr_metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    curr_metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    curr_metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
   
    
    
    return bce


def multi_loss_function(preds, target, curr_metrics):
    bce_weight = 0.5

    # find the bce 
    pred_1 = calc_bce(preds[0],target)
    pred_2 = calc_bce(preds[1],target)
    pred_3 = calc_bce(preds[2],target)
    pred_4 = calc_bce(preds[3],target)
    pred_5 = calc_bce(preds[4],target)

    # sum up all the bce losses and divide by 4 to get average across the 4 layers
    bce = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) 
    
    pred = torch.sigmoid(preds[0])
    
    # use the final layer output to calculate the dice score
    dice,dice_coeff = dc_loss(pred, target)

    # combo loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    curr_metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    curr_metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    curr_metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return bce

# evaluate validation set
def check_accuracy(loader, model, device="cuda"):

    model.eval()

    with torch.no_grad():       # we want to compare the mask and the predictions together / for binary
        epoch_samples = 0
        curr_metrics = defaultdict(float)

        for x, y in loader:
            
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            epoch_samples += x.size(0)

            # loss
            loss = calc_loss(pred, y, curr_metrics)
            
    val_dsc = curr_metrics['dice_coeff'] / epoch_samples
    val_bce = curr_metrics["bce"] / epoch_samples
    val_loss = curr_metrics['loss'] / epoch_samples
    

    metrics["val_loss"].append(val_loss)
    metrics["val_dice"].append(val_dsc)
    metrics["val_bce"].append(val_bce)
    
    
    
    print(f"Validation Loss: {val_loss} Validation Dice Score: {val_dsc} Validation BCE: {val_bce}")
    
    model.train()

    return val_dsc

def multi_check_accuracy(loader, model, device="cuda"):

    model.eval()

    with torch.no_grad():       # we want to compare the mask and the predictions together / for binary
        epoch_samples = 0
        curr_metrics = defaultdict(float)

        for x, y in loader:
            
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            epoch_samples += x.size(0)

            # loss
            loss = calc_loss(pred, y, curr_metrics)
            
    val_dsc = curr_metrics['dice_coeff'] / epoch_samples
    val_bce = curr_metrics["bce"] / epoch_samples
    val_loss = curr_metrics['loss'] / epoch_samples

    metrics["val_loss"].append(val_loss)
    metrics["val_dice"].append(val_dsc)
    metrics["val_bce"].append(val_bce)

    
    print(f"Validation Loss: {val_loss} Validation Dice Score: {val_dsc} Validation BCE: {val_bce}")
    
    model.train()

    return val_dsc
            


if __name__ == "__main__":
    torch.cuda.empty_cache()

    results = {}

    for lr in h_params["lr"]:
        for batch_size in h_params["batch_size"]:
            for epochs in h_params["epochs"]:

                print(f"current settings: {lr}_{epochs}_{batch_size} ")

                model = RPDNet(pretrained=True,freeze=False,fpa_block=True,respaths=True)
                print(DEVICE)

                # change this when u change model
                model.to(DEVICE)


                
                # Need to consider splitting the training set manually
                train_directory = "ISLES/TRAINING"
                val_directory = "ISLES/VALIDATION"

                modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT'] # remove ct image and try with only the other
                
                
                ### NEW STUFF ###
                directory = "ISLES/TRAINING"
                dataset = load_data(directory)

                train_data,val_data = train_test_split(dataset, test_size=0.3, train_size=0.7,random_state=20) # 30 before

                print( "Number of Patient Cases: ", len(dataset))
            

                # load our train and validation sets
                train_set = train_ISLES2018_loader(train_data, modalities)
                val_set = val_ISLES2018_loader(val_data, modalities)
                                

                # need to figure out how to import the data without issues with masks etc
                train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=NUM_WORKERS ,shuffle=True, pin_memory=True)
                valid_dl = DataLoader(val_set, batch_size=batch_size, num_workers=NUM_WORKERS ,shuffle=False, pin_memory=True)


                optimizer = optim.Adam(model.parameters(), lr)
            

                train_model(model, (train_dl, valid_dl),optimizer,epochs)


                results.update({f"{lr}_{epochs}_{batch_size}" : np.mean(metrics["val_dice"])})
                # reset metrics for next fold
                metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], "train_hd":[],"val_hd":[]}



                # reset model weights experiment
                del model, optimizer
                torch.cuda.empty_cache()
    
    print(results)



          
            

        




    
    
    

