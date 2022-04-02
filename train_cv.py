from collections import defaultdict
from math import gamma
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable

#from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
from patient_dataloader import train_ISLES2018_loader,val_ISLES2018_loader, load_data
#from patient_dataloader_mri import train_ISLES2018_loader,val_ISLES2018_loader, load_data
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, KFold
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR
import os


from scipy.spatial.distance import directed_hausdorff

# MODELS
from models.DSAnet import UNet_2D
from models.a_unet_model import UNet_Attention
from models.mult_res_unet import MultiResNet
from models.trans_unet import transUnet
from models.sa_unet import SAUNet_2D
from models.resUnet import SResUnet
from models.RPDnet import RPDNet

# Training Hyperparameters for replication of work:
# U-Net
# Attention U-Net
# MultiResNet
# TransUNet
# DSU-Net
# U-Net (with/without DS) with FPA
# U-Net (with/without DS) with MHSA
# U-Net (with/without DS) with ResNet-34 Backbone pre-trained / transfer learning
# U-Net (with/without DS) with ResNet-34 Backbone not pre-trained / no transfer learning
#


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

        #print(curr_metrics['hd'])
        # test validation dataset after each epoch
        train_loss = curr_metrics['loss'] / epoch_samples
        train_acc = curr_metrics["dice_coeff"] / epoch_samples
        train_bce = curr_metrics['bce'] / epoch_samples
        #train_precision = curr_metrics['precision'] / epoch_samples
        #train_recall = curr_metrics['recall'] / epoch_samples
        #train_hd = curr_metrics['hd'] / epoch_samples
       

        metrics["train_loss"].append(train_loss)
        metrics["train_dice"].append(train_acc)
        metrics["train_bce"].append(train_bce)
        #metrics["train_precision"].append(train_precision)
        #metrics["train_recall"].append(train_recall)
        #metrics["train_hd"].append(train_hd)
        
        
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
    #inputs = torch.sigmoid(inputs)       
        
    #flatten label and prediction tensors
    #inputs = inputs.view(-1)
    #targets = targets.view(-1)

    bceweight = torch.ones_like(targets)  +  20 * targets # create a weight for the bce that correlates to the size of the lesion
    
    
    #first compute binary cross-entropy 
    BCE = F.binary_cross_entropy_with_logits(pred, targets,weight=bceweight)
    BCE_EXP = torch.exp(-BCE)
    focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
    return focal_loss
    
def hausdorff(preds, targets):
    '''
    :param probs: 4 x 1 x 256 x 256
    :param target:
    :return:
    '''

    hd = 0.0

    # calculate the hausdorff distance for each individual image per batch
    for i in range(0,preds.shape[0]):
        output = preds[i, : , : , :].detach().cpu().squeeze().numpy()
        segment = targets[i, : , : , :].detach().cpu().squeeze().numpy()
        dh1 = directed_hausdorff(output, segment)[0]
        dh2 = directed_hausdorff(segment, output)[0]
        hd += np.max([dh1, dh2])

    # return the total hd for this batch of images
    return hd 


# separate this bit and move the dc loss function into the train.py file...
# calculate weighted loss
def calc_loss(pred, target, curr_metrics):

    bce_weight = 0.5
    
    bce = calc_bce(pred,target)
    
    pred = torch.sigmoid(pred)

    
    #hd_loss = hausdorff(pred,target)
    
    #_, precision, recall = f1_score(pred,target)
    
    dice,dice_coeff = dc_loss(pred, target)

    # combo loss
    loss = bce * bce_weight + dice * (1 - bce_weight)

    curr_metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    curr_metrics['dice_coeff'] += dice_coeff.data.cpu().numpy() * target.size(0)
    curr_metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    #curr_metrics['precision'] += precision.data.cpu().numpy() * target.size(0)
    #curr_metrics['recall'] += recall.data.cpu().numpy() * target.size(0)
    #curr_metrics['hd'] += hd_loss 
    
    
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
    bce = (pred_1 + pred_2 + pred_3 + pred_4 + pred_5) / len(preds)
    
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
    #val_precision = curr_metrics['precision'] / epoch_samples
    #val_recall = curr_metrics['recall'] / epoch_samples
    val_hd = curr_metrics['hd'] / epoch_samples
    

    metrics["val_loss"].append(val_loss)
    metrics["val_dice"].append(val_dsc)
    metrics["val_bce"].append(val_bce)
    #metrics["val_precision"].append(val_precision)
    #metrics["val_recall"].append(val_recall)
    #metrics["val_hd"].append(val_hd)
    
    
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

    plot_graph = False

    # define the number of folds/epochs per fold
    k_folds = 5
    num_epochs = 1

    results = {"train_dice" : [],"val_dice" : []}

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

        # define model
        #model = UNet_2D(in_channels=1,fpa_block=True, sa=False,deep_supervision=DEEP_SUPERVISION, mhca=False) # make sure to change the number of channels in the unet model file
        model = RPDNet(pretrained=True,freeze=False,fpa_block=True,respaths=True)
        
        # send model to device (gpu)
        model.to(DEVICE)

        # define dataloaders
        train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=True, pin_memory=True)
        valid_dl = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS ,shuffle=False, pin_memory=True)

        
        # define optimizer
        optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=60, gamma=0.01)

        # training
        train_model(model, (train_dl, valid_dl),optimizer,NUM_EPOCHS,scheduler=scheduler)

        # find mean dice coefficient of this fold and reset all data for next fold

        mean_train_dice = np.mean(metrics["train_dice"])
        mean_val_dice = np.mean(metrics["val_dice"])

        results["train_dice"].append(mean_train_dice)
        results["val_dice"].append(mean_val_dice)

        

        # optional plot function

        if plot_graph:
            
            fig, (ax1, ax2) = plt.subplots(1, 2)
        
            ax1.plot(metrics["train_loss"],label="training loss")
            ax1.plot(metrics["val_loss"], label="validation loss")
            ax1.set_ylabel("Loss")
            ax1.set_xlabel("Epochs")
            ax1.set_title("Loss")
            ax1.legend(loc="upper right")
            
            ax2.plot(metrics["train_dice"],label="training dice")
            ax2.plot(metrics["val_dice"], label="validation dice")
            ax2.set_title("Dice Score")
            ax2.set_ylabel("Dice Coefficient")
            ax2.set_xlabel("Epochs")
            ax2.legend(loc="upper right")

            fig.savefig(f"fold_{fold}_ResNet34_trained.jpg")
        
        # reset metrics for next fold
        metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], "train_hd":[],"val_hd":[]}

        # reset model weights per fold
        del model, optimizer
        torch.cuda.empty_cache()
    
    # view final results
    print(np.mean(results["train_dice"]))
    print(np.mean(results["val_dice"]))
            

        


    """
    # load pretrained vit model for weight extraction
    m1 = timm.create_model('vit_base_patch16_384',pretrained='True')

    #for name,param in m1.named_parameters():
     #   print(name)
    
    # declare model
    model = transUnet(p=0.2,attn_p=0.2)

    # create model weight dict
    transunet_model_dict = model.state_dict()

    model.load_from(weights=np.load(imagenet21k_R50+ViT-B_16.npz))

    # load the model weights only for the specific parts like ViT
    pretrained_dict = {k: v for k, v in m1.state_dict().items() if k in transunet_model_dict}

    # update weight dict
    transunet_model_dict.update(pretrained_dict)

    # load weights
    model.load_state_dict(transunet_model_dict)

    # need to freeze the ViT weights specifically
    for params in model.blocks.children():
        for param in params.parameters():
            param.requires_grad = False
    """

    #model.blocks[3].attn.proj.weight.requires_grad = True

    # view summary of model
    #summary(model, input_size=(4,5,256,256))

    #print(model.blocks[3].attn.proj.bias)

    #print(model.vit.mlp_head.weight)
    #print(model.vit.transformer.layers[0])

    
    #for name,param in model.named_parameters():
    #   print(name,param)

    
    
    

