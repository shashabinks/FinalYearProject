from collections import defaultdict

import torch

from torch.nn import functional as F
import numpy as np

from patient_dataloader_aug import train_ISLES2018_loader,val_ISLES2018_loader, load_data
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import  KFold
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR
import os



# MODELS

#from a_unet_model import UNet_Attention 
from models.DSAnet import UNet_2D
from models.RPDnet import RPDNet
from models.unet_pp import NestedUNet

from models.swin_transformer.segmentors.encoder_decoder import EncoderDecoder as Swin

# hyperparameters
LEARNING_RATE = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 35+1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
TRANSFORMER = False
DEEP_SUPERVISION = False
MODEL = Swin()



metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], "train_jaccard":[], "val_jaccard":[]
            , "train_precision":[], "val_precision":[], "train_recall":[], "val_recall":[]}

fold_metrics = {}

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
        sample = 0
        epoch_samples = 0
        curr_metrics = defaultdict(float)

        # iterate through the batches
        for i, data in enumerate(loaders[0]):
            
            
            # put images into devices
            train_image, ground_truth = data[0].to(DEVICE), data[1].to(DEVICE)

            # reset gradients to zero in prep for next batch
            optimizer.zero_grad()
    
            epoch_samples += train_image.size(0) # batch num
            sample += 1
            
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
        train_jaccard = curr_metrics['jaccard'] / epoch_samples
        train_precision = curr_metrics['precision'] / epoch_samples
        train_recall = curr_metrics['recall'] / epoch_samples

        metrics["train_loss"].append(train_loss)
        metrics["train_dice"].append(train_acc)
        metrics["train_bce"].append(train_bce)
        metrics["train_precision"].append(train_precision)
        metrics["train_recall"].append(train_recall)
        metrics["train_jaccard"].append(train_jaccard)
        
        print(f"Epoch: {epoch}")
        print(f"Train Loss: {train_loss} Train Dice Score: {train_acc} Train BCE: {train_bce} Train Jaccard: {train_jaccard} Train Precision: {train_precision} Train Recall: {train_recall}")
        
        epoch_val_acc = 0.0

        if DEEP_SUPERVISION:
            epoch_val_acc = multi_check_accuracy(loaders[1], model, device=DEVICE)

        else:
            epoch_val_acc = check_accuracy(loaders[1], model, device=DEVICE)

        if epoch_val_acc > best_score:
            print("Best Accuracy so far!, Saving model...")
            best_score = epoch_val_acc
            torch.save(model.state_dict(), os.path.join("", "best_weights.pth"))


        
        # view images after all epochs
        if epoch % (NUM_EPOCHS-1) == 0:
            if DEEP_SUPERVISION:
                view_images_multi(loaders[1], model, device=DEVICE)

            
            if epoch == NUM_EPOCHS - 1:
                print("Final Epoch!, Saving model...")

                torch.save(model.state_dict(), os.path.join("", "final_weights.pth"))
        
        if scheduler:
            scheduler.step()


def view_images_multi(loader, model, device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)[0])
            #preds = (preds > 5.0e-4).float()
        
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


def focal_loss(inputs, targets, smooth=1, alpha=0.7, beta=0.3, gamma=0.75):
    

    #comment out if your model contains a sigmoid or equivalent activation layer
    #inputs = F.sigmoid(inputs)       
    
    #flatten label and prediction tensors
    inputs = inputs.contiguous()
    targets = targets.contiguous()
    
    #True Positives, False Positives & False Negatives
    TP = (inputs * targets).sum(dim=2).sum(dim=2)    
    FP = ((1-targets) * inputs).sum(dim=2).sum(dim=2) 
    FN = (targets * (1-inputs)).sum(dim=2).sum(dim=2) 
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma
                    
    return FocalTversky
    

def jaccard_coeff(inputs, targets):

    """
    inputs = inputs.contiguous()
    targets = targets.contiguous()

    smooth = 1.


    intersection = (inputs * targets).sum(dim=2).sum(dim=2)  
    union = (inputs.sum(dim=2).sum(dim=2)  + targets.sum(dim=2).sum(dim=2) ) - intersection

    return (intersection / (union + smooth)).mean()

    
    """

    smooth = 1e-5

    targets = (targets * 255).float() # convert all mask values to 1 or 0
    inputs = (inputs > 5e-4).float() # convert all prediction values between 1 or 0

    output = inputs.view(-1)
    target = targets.view(-1)

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    jaccard = tp / (tp + fp + fn + smooth)
    return jaccard

def precision_and_recall(inputs , targets):

    smooth = 1e-5

    targets = (targets * 255).float() # convert all mask values to 1 or 0
    inputs = (inputs > 5e-4).float() # convert all prediction values between 1 or 0, set this to 0.0005 as it seems there is an issue with the values being to small

    output = inputs.view(-1)
    target = targets.view(-1)

    tp = torch.sum(output * target)  # TP
    fp = torch.sum(output * (1 - target))  # FP
    fn = torch.sum((1 - output) * target)  # FN
    tn = torch.sum((1 - output) * (1 - target))  # TN

    jaccard = tp / (tp + fp + fn + smooth)

    precision = tp/(tp + fp + smooth)
    recall = tp/(tp + fn + smooth)

    return precision, recall

# separate this bit and move the dc loss function into the train.py file...
# calculate weighted loss
def calc_loss(pred, target, curr_metrics):

    bce_weight = 0.5
    
    bce = calc_bce(pred,target)

    # sigmoid activation
    pred = torch.sigmoid(pred)

    #focal = focal_loss(pred,target)

    precision, recall = precision_and_recall(pred,target)

    jaccard = jaccard_coeff(pred,target)
    
    dice,dice_coeff = dc_loss(pred, target)

    # combo loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    curr_metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    curr_metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    curr_metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    curr_metrics['precision'] += precision.data.cpu().numpy() * target.size(0)
    curr_metrics['recall'] += recall.data.cpu().numpy() * target.size(0)
    curr_metrics['jaccard'] += jaccard.data.cpu().numpy() * target.size(0)
    
    return bce


def multi_loss_function(preds, target, curr_metrics):
    bce_weight = 0.5

    # find the bce 
    pred_1 = calc_bce(preds[0],target)
    pred_2 = calc_bce(preds[1],target)
    pred_3 = calc_bce(preds[2],target)
    pred_4 = calc_bce(preds[3],target)

    # sum up all the bce losses and divide by 4 to get average across the 4 layers
    bce = (pred_1 + pred_2 + pred_3 + pred_4) 
    
    pred = torch.sigmoid(preds[0])

    precision, recall = precision_and_recall(pred,target)

    jaccard = jaccard_coeff(pred,target)
    
    # use the final layer output to calculate the dice score
    dice,dice_coeff = dc_loss(pred, target)

    # combo loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    curr_metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    curr_metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    curr_metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    curr_metrics['precision'] += precision.data.cpu().numpy() * target.size(0)
    curr_metrics['recall'] += recall.data.cpu().numpy() * target.size(0)
    curr_metrics['jaccard'] += jaccard.data.cpu().numpy() * target.size(0)
    
    return bce

# evaluate validation set
def check_accuracy(loader, model, device="cuda"):

    model.eval()

    with torch.no_grad():       # we want to compare the mask and the predictions together / for binary
        epoch_samples = 0
        curr_metrics = defaultdict(float)
        sample = 0

        for x, y in loader:
            
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            epoch_samples += x.size(0)
            sample += 1

            # loss
            loss = calc_loss(pred, y, curr_metrics)
            
    val_dsc = curr_metrics['dice_coeff'] / epoch_samples
    val_bce = curr_metrics["bce"] / epoch_samples
    val_loss = curr_metrics['loss'] / epoch_samples
  
    val_jaccard = curr_metrics['jaccard'] / epoch_samples
    val_precision = curr_metrics['precision'] / epoch_samples
    val_recall = curr_metrics['recall'] / epoch_samples

    metrics["val_loss"].append(val_loss)
    metrics["val_dice"].append(val_dsc)
    metrics["val_bce"].append(val_bce)
    metrics["val_precision"].append(val_precision)
    metrics["val_recall"].append(val_recall)
    metrics["val_jaccard"].append(val_jaccard)
    

    
    print(f"Validation Loss: {val_loss} Validation Dice Score: {val_dsc} Validation BCE: {val_bce} Val Jaccard: {val_jaccard} Val Precision: {val_precision} Train Recall: {val_recall}")
    
    model.train()

    return val_dsc

def multi_check_accuracy(loader, model, device="cuda"):

    model.eval()

    with torch.no_grad():       # we want to compare the mask and the predictions together / for binary
        epoch_samples = 0
        curr_metrics = defaultdict(float)
        sample = 0

        for x, y in loader:
            
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            epoch_samples += x.size(0)
            sample += 1

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

    plot_graph = False

    # define the number of folds/epochs per fold
    k_folds = 5
    num_epochs = 1

    results = {"train_dice" : [],"val_dice" : [], "train_precision":[], "val_precision":[], "train_jaccard":[], "val_jaccard":[], "train_recall":[], "val_recall":[], "train_loss":[], "val_loss":[]}

    torch.manual_seed(42)

    # load data
    directory = "ISLES/TRAINING"
    dataset = load_data(directory)

    # define modalities
    modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']

    # define k-fold cross validator
    kfold = KFold(n_splits=k_folds, shuffle=True)

    for fold, (train_ids,test_ids) in enumerate(kfold.split(dataset)):

        
        train_list = []
        test_list = []

        print(f'Fold {fold}')

        # create list of indexes for each split
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # loop through each id and create the test/valid set
        for i in list(train_subsampler):
            train_list.append(dataset[i])

        for i in list(test_subsampler):
            test_list.append(dataset[i])
        
        train_set = train_ISLES2018_loader(train_list, modalities)
        print("Loaded Training Data...")
        val_set = val_ISLES2018_loader(test_list, modalities)
        print("Loaded Validation Data...")

        print(len(train_set))
        print(len(val_set))

        model = MODEL
        
        # send model to device (gpu)
        model.to(DEVICE)

        # define dataloaders
        train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=True, pin_memory=True)
        valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=False, pin_memory=True)

        
        # define optimizer
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

        # training
        train_model(model, (train_dl, valid_dl),optimizer,NUM_EPOCHS,scheduler=scheduler)

        # find mean dice coefficient of this fold and reset all data for next fold

        mean_train_dice = np.mean(metrics["train_dice"])
        mean_val_dice = np.mean(metrics["val_dice"])

        mean_train_loss = np.mean(metrics["train_bce"])
        mean_val_loss = np.mean(metrics["val_bce"])

        mean_train_jaccard = np.mean(metrics["train_jaccard"])
        mean_val_jaccard = np.mean(metrics["val_jaccard"])

        mean_train_precision = np.mean(metrics["train_precision"])
        mean_val_precision = np.mean(metrics["val_precision"])

        mean_train_recall = np.mean(metrics["train_recall"])
        mean_val_recall = np.mean(metrics["val_recall"])

        results["train_dice"].append(mean_train_dice)
        results["val_dice"].append(mean_val_dice)

        results["train_loss"].append(mean_train_loss)
        results["val_loss"].append(mean_val_loss)

        results["train_jaccard"].append(mean_train_jaccard)
        results["val_jaccard"].append(mean_val_jaccard)

        results["train_recall"].append(mean_train_recall)
        results["val_recall"].append(mean_val_recall)

        results["train_precision"].append(mean_train_precision)
        results["val_precision"].append(mean_val_precision)

        

        # optional plot function

        if plot_graph:
            
            fig, (ax1, ax2) = plt.subplots(1, 2)
        
            ax1.plot(metrics["train_loss"],label="Training Loss")
            ax1.plot(metrics["val_loss"], label="Validation Loss")
            ax1.set_title("Loss")
            ax1.legend(loc="upper right")
            
            ax2.plot(metrics["train_dice"],label="Training Dice")
            ax2.plot(metrics["val_dice"], label="Validation Dice")
            ax2.set_title("Dice Score")
            ax2.legend(loc="upper right")

            ax1.set_xlabel("Epochs")
            ax2.set_xlabel("Epochs")

            fig.savefig(f"fold_{fold}_aa_unet_untrained.jpg")

        
        fold_metrics.update({f"fold_{fold}": metrics["val_dice"]})
        
        # reset metrics for next fold
        metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], 
            "train_precision":[],"val_precision":[], "train_recall":[], "val_recall":[], "train_jaccard":[], "val_jaccard":[]}

        # reset model weights per fold
        reset_weights(model)
        del model, optimizer
        torch.cuda.empty_cache()
        
        
    
    # view final results
    print(np.mean(results["train_dice"]))
    print(np.mean(results["val_dice"]))

    print(np.mean(results["train_loss"]))
    print(np.mean(results["val_loss"]))

    print(np.mean(results["train_precision"]))
    print(np.mean(results["val_precision"]))

    print(np.mean(results["train_jaccard"]))
    print(np.mean(results["val_jaccard"]))

    print(np.mean(results["train_recall"]))
    print(np.mean(results["val_recall"]))

    # view box plot

    data = [fold_metrics["fold_0"],fold_metrics["fold_1"],fold_metrics["fold_2"],fold_metrics["fold_3"],fold_metrics["fold_4"]]
    fig = plt.figure(figsize =(10, 7))
    
    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel("Folds")
    ax.set_ylabel("DSC")
    
    # Creating plot
    bp = ax.boxplot(data)
    
    # show plot
    plt.show()

    
            

        



    
    
    

