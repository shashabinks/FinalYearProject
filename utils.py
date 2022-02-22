from cProfile import label
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from zmq import device
from model import UNet

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

metrics = {}
# calculate correlation between the pred image and the label
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1.):

        
        
        inputs = inputs.contiguous()

        targets = targets.contiguous()
        
        intersection = (inputs * targets).sum(dim=2).sum(dim=2)                              
        dice = (2.*intersection + smooth)/(inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)  

        loss = 1 - dice
        
        return loss.mean(),dice.mean() # output loss

def dc_loss(inputs,targets,smooth=1.):
    inputs = inputs.contiguous()

    targets = targets.contiguous()
    
    intersection = (inputs * targets).sum(dim=2).sum(dim=2)                              
    dice = (2.*intersection + smooth)/(inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)  

    loss = 1 - dice
    
    return loss.mean(),dice.mean() # output loss

def calc_loss(pred=None, target=None, bce_weight=0.5):

    bceweight = torch.ones_like(target)  +  20 * target
    bce = F.binary_cross_entropy_with_logits(pred, target, weight= bceweight)
    
    pred = torch.sigmoid(pred)
    dice_loss = DiceLoss()
    dice,dice_coeff = dice_loss(pred, target)

    #print(f"Train Accuracy:{dice_coeff}")

    loss = bce * bce_weight + dice * (1 - bce_weight)
    
   
    
    return loss


def check_accuracy(loader, model, device="cuda"):
    dice_scores = []
    dice_loss = []
    bce_loss = []
    model.eval()

    with torch.no_grad():       # we want to compare the mask and the predictions together / for binary
        for x, y in loader:
            #plt.imshow(y[0].squeeze(0), cmap="gray")
            #plt.show()
            x = x.to(device)
            y = y.to(device)
            
            # loss
            preds = model(x)
            loss = calc_loss(preds, y)
            dice_loss.append(loss)

            # dice score
            preds = torch.sigmoid(preds)
            _,coeff = dc_loss(preds,y) # change the loss to the weighted loss
            dice_scores.append(coeff)
            
    
    overall_dsc = torch.stack(dice_scores).mean().item()
    overall_loss = torch.stack(dice_loss).mean().item()
    
    print(f"Validation Loss: {overall_loss} Validation Acc: {overall_dsc}")
    model.train()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
        
        #plt.imshow(y[0].squeeze(0), cmap="gray")
        #plt.show()

        #print(y[0].shape)
        #torchvision.utils.save_image(y, f"{folder}{idx}.png")
        #torchvision.utils.save_image(preds , f"{folder}/pred_{idx}.png")
        #print(idx)
        if idx == 1 or idx == 2 or idx == 4 or idx == 6 or idx == 8:
            f, (ax2, ax3) = plt.subplots(1, 2, figsize=(10,20))
            
            ax2.imshow(preds[0].cpu().squeeze().numpy(), cmap = 'gray',label="Prediction")
            ax3.imshow(y[0].squeeze(0), cmap= 'gray', label="Mask")
            #plt.legend()
            plt.show()

    model.train()