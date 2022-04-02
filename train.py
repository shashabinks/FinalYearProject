from collections import defaultdict
from math import gamma
import torch
import torch.nn as nn
from torch.nn import functional as F
import timm
from torch.autograd import Variable


from torchsummary import summary
#from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
from patient_dataloader_aug import train_ISLES2018_loader,val_ISLES2018_loader, load_data
#from patient_dataloader_mri import train_ISLES2018_loader,val_ISLES2018_loader, load_data
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import save_predictions_as_imgs
from torch.optim.lr_scheduler import StepLR
import os
import numpy as np
from scipy.spatial.distance import directed_hausdorff

import segmentation_models_pytorch as smp


# MODELS
from models.DSAnet import UNet_2D
from models.a_unet_model import UNet_Attention
from models.mult_res_unet import MultiResNet
from models.trans_unet import transUnet
from models.sa_unet import SAUNet_2D
from models.resUnet import SResUnet
#from models.RPDnet import RPDNet
from models.transunet.vit_seg_modeling import VisionTransformer
from models.transunet.vit_seg_modeling import CONFIGS
from models.final_net import RPDNet

from loss_func import BinaryMetrics
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
NUM_EPOCHS = 100 + 1
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
TRANSFORMER = False
DEEP_SUPERVISION = False


metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[], "train_jaccard":[], "val_jaccard":[]
            , "train_precision":[], "val_precision":[], "train_recall":[], "val_recall":[]}
      
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
            torch.save(model.state_dict(), os.path.join("Weights/", "best_weights.pth"))


        
        # view images after all epochs
        if epoch % (NUM_EPOCHS-1) == 0:
            if DEEP_SUPERVISION:
                view_images_multi(loaders[1], model, device=DEVICE)

            else:
                save_predictions_as_imgs(loaders[1], model, folder="saved_images/", device=DEVICE)

            if epoch == NUM_EPOCHS - 1:
                print("Final Epoch!, Saving model...")

                torch.save(model.state_dict(), os.path.join("Weights/", "final_weights.pth"))
        
        if scheduler:
            scheduler.step()


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
    
def compute_hausdorff(preds, targets):
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

def jaccard_coeff(inputs, targets):
    eps = 1.0

    inputs = inputs.contiguous()
    targets = targets.contiguous()

    intersection = (inputs * targets).sum(dim=2).sum(dim=2)   
    union = (inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2)) - intersection

    return (intersection / (union + eps)).mean()

def precision_and_recall(input , target):

    eps = 1e-5

    input = input.view(-1)
    target = target.view(-1).float()

    tp = torch.sum(input * target)  # TP
    fp = torch.sum(input * (1 - target))  # FP
    fn = torch.sum((1 - input) * target)  # FN
    tn = torch.sum((1 - input) * (1 - target))  # TN

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return precision, recall

# separate this bit and move the dc loss function into the train.py file...
# calculate weighted loss
def calc_loss(pred, target, curr_metrics):

    bce_weight = 0.5
    
    bce = calc_bce(pred,target)

    
    
    # sigmoid activation
    pred = torch.sigmoid(pred)

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

    """
    # load pretrained vit model for weight extraction
    m1 = timm.create_model('vit_base_patch16_384',pretrained='True')
    m2 = timm.create_model('resnet50',pretrained='True')

    #for name,param in m1.named_parameters():
     #   print(name)
    
    # declare model
    model = transUnet()

    #summary(model, input_data=(5,256,256))

    # create model weight dict
    transunet_model_dict = model.state_dict()

    #model.load_from(weights=np.load("imagenet21k_R50+ViT-B_16.npz"))

    # load the model weights only for the specific parts like ViT
    pretrained_dict = {k: v for k, v in m1.state_dict().items() if k in transunet_model_dict}
    pretrained_dict_1 = {k: v for k, v in m2.state_dict().items() if k in transunet_model_dict}

    # update weight dict
    transunet_model_dict.update(pretrained_dict)
    transunet_model_dict.update(pretrained_dict_1)

    # load weights
    model.load_state_dict(transunet_model_dict)

    
    
    #for name,param in model.named_parameters():
    #   print(name,param)
    """


    #model = UNet_2D(in_channels=1,fpa_block=True, sa=False,deep_supervision=DEEP_SUPERVISION, mhca=False) # make sure to change the number of channels in the unet model file
    
    
    model = RPDNet(pretrained=True,freeze=False,fpa_block=True,respaths=True,mhca=True)
    
    #config_vt = CONFIGS["R50-ViT-B_16"]
    #model = VisionTransformer(config_vt, img_size=256,num_classes=1)
    #model.load_from(weights=np.load(config_vt.pretrained_path))
    print(DEVICE)

    # change this when u change model
    model.to(DEVICE)
 
    # Need to consider splitting the training set manually
    train_directory = "ISLES/TRAINING"
    val_directory = "ISLES/VALIDATION"

    modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT'] # remove ct image and try with only the other
    #modalities = ['OT', 'CT_4DPWI'] # remove ct image and try with only the other
    
    ### NEW STUFF ###
    directory = "ISLES/TRAINING"
    dataset = load_data(directory)

    train_data,val_data = train_test_split(dataset, test_size=0.3, train_size=0.7,random_state=20) # 30 before

    print( "Number of Patient Cases: ", len(dataset))
    

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


    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.01)

    train_model(model, (train_dl, valid_dl),optimizer,NUM_EPOCHS,scheduler=scheduler)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(metrics["train_loss"],label="training bce loss")
    ax1.plot(metrics["val_loss"], label="validation bce loss")
    ax1.set_title("Loss")
    ax1.legend(loc="upper right")
    
    ax2.plot(metrics["train_dice"],label="training dice")
    ax2.plot(metrics["val_dice"], label="validation dice")
    ax2.set_title("Dice Score")
    ax2.legend(loc="upper right")

    plt.legend()
    plt.show()
    
    

