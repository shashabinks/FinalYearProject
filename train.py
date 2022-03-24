from collections import defaultdict
import torch
import torch.nn as nn
from torch.nn import functional as F

#from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
from patient_dataloader import train_ISLES2018_loader,val_ISLES2018_loader, load_data
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

from torchinfo import summary

# MODELS
from models.DSAnet import UNet_2D
from models.a_unet_model import Attention_block, UNet_Attention
from models.mm_unet_four import DMM_Unet_4
from models.mm_unet import DMM_Unet
from models.mult_res_unet import MultiResNet
from models.pan_unet import PAN_Unet
from models.unet_pp import PP_Unet
from models.unet_cbam import Unet_CBAM
from models.custom_unet import TDMM_Unet_4
from models.trans_unet import transUnet
from models.t_unet import TransUnet
from models.unet_aspp import UNet_ASPP
from models.mptrans_unet import MPT_Net
from models.mma_unet import MMA_Net
from models.samr_net import MultiResBlock, SAMRnet
from models.sub_unet import SubUNet
from models.sa_unet import SAUNet_2D
from models.resUnet import SResUnet


# hyperparameters
LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100+1
NUM_WORKERS = 2
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
TRANSFORMER = False
DEEP_SUPERVISION = True


metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[]}
      
# define training function
def train_model(model,loaders,optimizer,num_of_epochs,scheduler=None):

    best_score = -1.0
    
    # iterate through the epochs
    for epoch in range(0,num_of_epochs):

        epoch_samples = 0
        curr_metrics = defaultdict(float)

        loss = None

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

# weighted due to class imbalance
def calc_bce(pred=None, target=None):
    bceweight = torch.ones_like(target)  +  25 * target # create a weight for the bce that correlates to the size of the lesion
    bce = nn.BCEWithLogitsLoss(weight = bceweight) # the size of the lesions are small therefore it is important to use this
    bce_loss = bce(pred, target)
    return bce_loss

# used in PAN paper
def focal_loss(alpha,gamma,bce_loss):
    pt = torch.exp(-bce_loss)
    focal_loss = (alpha * (1-pt)**gamma * bce_loss)
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

    # sum up all the bce losses and divide by 4 to get average across the 4 layers
    bce = (pred_1 + pred_2 + pred_3 + pred_4) / 4.0
    
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


def identify_axis(shape):
    # Three dimensional
    if len(shape) == 5 : return [1,2,3]

    # Two dimensional
    elif len(shape) == 4 : return [1,2]
    
    # Exception - Unknown
    else : raise ValueError('Metric: Shape of tensor is neither 2D or 3D.')


class SymmetricFocalLoss(nn.Module):
    """
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=2., epsilon=1e-07):
        super(SymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        axis = identify_axis(y_true.size())  
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)

        # Calculate losses separately for each class
        back_ce = torch.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = torch.pow(1 - y_pred[:,:,:,1], self.gamma) * cross_entropy[:,:,:,1]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class AsymmetricFocalLoss(nn.Module):
    """For Imbalanced datasets
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.25
    gamma : float, optional
        Focal Tversky loss' focal parameter controls degree of down-weighting of easy examples, by default 2.0
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.25, gamma=2., epsilon=1e-07):
        super(AsymmetricFocalLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):

        axis = identify_axis(y_true.size())  
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        cross_entropy = -y_true * torch.log(y_pred)
        
	# Calculate losses separately for each class, only suppressing background class
        back_ce = torch.pow(1 - y_pred[:,:,:,0], self.gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - self.delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = self.delta * fore_ce

        loss = torch.mean(torch.sum(torch.stack([back_ce, fore_ce], axis=-1), axis=-1))

        return loss


class SymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())
        
        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1-dice_class[:,0]) * torch.pow(1-dice_class[:,0], -self.gamma)
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss


class AsymmetricFocalTverskyLoss(nn.Module):
    """This is the implementation for binary segmentation.
    Parameters
    ----------
    delta : float, optional
        controls weight given to false positive and false negatives, by default 0.7
    gamma : float, optional
        focal parameter controls degree of down-weighting of easy examples, by default 0.75
    smooth : float, optional
        smooithing constant to prevent division by 0 errors, by default 0.000001
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, delta=0.7, gamma=0.75, epsilon=1e-07):
        super(AsymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Clip values to prevent division by zero error
        y_pred = torch.clamp(y_pred, self.epsilon, 1. - self.epsilon)
        axis = identify_axis(y_true.size())

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = torch.sum(y_true * y_pred, axis=axis)
        fn = torch.sum(y_true * (1-y_pred), axis=axis)
        fp = torch.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + self.epsilon)/(tp + self.delta*fn + (1-self.delta)*fp + self.epsilon)

        # Calculate losses separately for each class, only enhancing foreground class
        back_dice = (1-dice_class[:,0]) 
        fore_dice = (1-dice_class[:,1]) * torch.pow(1-dice_class[:,1], -self.gamma) 

        # Average class scores
        loss = torch.mean(torch.stack([back_dice,fore_dice], axis=-1))
        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to symmetric Focal Tversky loss and symmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      symmetric_ftl = SymmetricUnifiedFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
      symmetric_fl = SymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)
      if self.weight is not None:
        return (self.weight * symmetric_ftl) + ((1-self.weight) * symmetric_fl)  
      else:
        return symmetric_ftl + symmetric_fl


class AsymmetricUnifiedFocalLoss(nn.Module):
    """The Unified Focal loss is a new compound loss function that unifies Dice-based and cross entropy-based loss functions into a single framework.
    Parameters
    ----------
    weight : float, optional
        represents lambda parameter and controls weight given to asymmetric Focal Tversky loss and asymmetric Focal loss, by default 0.5
    delta : float, optional
        controls weight given to each class, by default 0.6
    gamma : float, optional
        focal parameter controls the degree of background suppression and foreground enhancement, by default 0.5
    epsilon : float, optional
        clip values to prevent division by zero error
    """
    def __init__(self, weight=0.5, delta=0.6, gamma=0.2):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, y_pred, y_true):
      # Obtain Asymmetric Focal Tversky loss
      asymmetric_ftl = AsymmetricFocalTverskyLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

      # Obtain Asymmetric Focal loss
      asymmetric_fl = AsymmetricFocalLoss(delta=self.delta, gamma=self.gamma)(y_pred, y_true)

      # Return weighted sum of Asymmetrical Focal loss and Asymmetric Focal Tversky loss
      if self.weight is not None:
        return (self.weight * asymmetric_ftl) + ((1-self.weight) * asymmetric_fl)  
      else:
        return asymmetric_ftl + asymmetric_fl
            


if __name__ == "__main__":
    torch.cuda.empty_cache()

    """
    # load pretrained vit model for weight extraction
    m1 = timm.create_model('vit_base_patch16_384',pretrained='True')

    #for name,param in m1.named_parameters():
     #   print(name)
    
    # declare model
    model = transUnet(p=0.2,attn_p=0.2)

    # create model weight dict
    transunet_model_dict = model.state_dict()

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

    model = UNet_2D(fpa_block=True, deep_supervision=True) # make sure to change the number of channels in the unet model file
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


    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=0.001)
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.3)

    train_model(model, (train_dl, valid_dl),optimizer,NUM_EPOCHS)

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
    
    

