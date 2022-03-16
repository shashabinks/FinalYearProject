from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

#from datasetloader import train_ISLES2018_loader,val_ISLES2018_loader
from patient_dataloader import train_ISLES2018_loader,val_ISLES2018_loader, load_data
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from utils import save_predictions_as_imgs
from torch.optim.lr_scheduler import StepLR
import timm

# MODELS
from models.unet_model import UNet_2D
from models.a_unet_model import Attention_block, UNet_Attention
from models.mm_unet_four import DMM_Unet_4
from models.mm_unet import DMM_Unet
from models.mult_res_unet import MultiResNet
from models.pan_unet import PAN_Unet
from models.unet_pp import PP_Unet
from models.unet_cbam import Unet_CBAM
from models.custom_unet import RPAN_Unet
from models.trans_unet import transUnet
from models.unet.unet_transformer.unet import TransUnet


# hyperparameters
LEARNING_RATE = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100+1
NUM_WORKERS = 4
IMAGE_HEIGHT = 256 
IMAGE_WIDTH = 256  
PIN_MEMORY = True
LOAD_MODEL = False
TRANSFORMER = False


metrics = {"train_bce":[],"val_bce":[],"train_dice":[],"val_dice":[],"train_loss":[],"val_loss":[]}
      

def train_model(model,loaders,optimizer,num_of_epochs,scheduler=None):
    
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
        
        check_accuracy(loaders[1], model, device=DEVICE)

        # view images after 100 epochs
        if epoch % (NUM_EPOCHS-1) == 0:
            save_predictions_as_imgs(loaders[1], model, folder="saved_images/", device=DEVICE)
        
        
# calculate dice coefficient/loss
def dc_loss(inputs,targets,smooth=1.):
    inputs = inputs.contiguous()

    targets = targets.contiguous()
    
    intersection = (inputs * targets).sum(dim=2).sum(dim=2)                              
    dice = (2.*intersection + smooth)/(inputs.sum(dim=2).sum(dim=2) + targets.sum(dim=2).sum(dim=2) + smooth)  

    loss = 1 - dice
    
    return loss.mean(),dice.mean() # output loss


def calc_bce(pred=None, target=None):
    bceweight = torch.ones_like(target)  +  20 * target # create a weight for the bce that correlates to the size of the lesion
    bce = nn.BCEWithLogitsLoss(weight = bceweight) # the size of the lesions are small therefore it is important to use this
    bce_loss = bce(pred, target)
    return bce_loss

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
            


if __name__ == "__main__":
    torch.cuda.empty_cache()

    m1 = timm.create_model('vit_base_patch16_384',pretrained='True')
    #model = transUnet(p=0.2,attn_p=0.2)
    model = TransUnet(in_channels=5,img_dim=256,vit_blocks=1,vit_dim_linear_mhsa_block=512,classes=1)
    transunet_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in m1.state_dict().items() if k in transunet_model_dict}
    transunet_model_dict.update(pretrained_dict)
    model.load_state_dict(transunet_model_dict)



    #model = UNet_2D() # make sure to change the number of channels in the unet model file
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

    train_data,val_data = train_test_split(dataset, test_size=0.3, train_size=0.7,random_state=30) # 30 before

    print( "Number of Patient Cases: ", len(dataset))
    #print(dataset[0])
    

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
    #scheduler = StepLR(optimizer, step_size=8, gamma=0.3)

    train_model(model, (train_dl, valid_dl),optimizer,NUM_EPOCHS)

    #plt.plot(metrics["train_loss"], label="training bce loss")
    #plt.plot(metrics["val_loss"], label="validation bce loss")
    #plt.xlabel("Num of Epochs")
    #plt.ylabel("")
    #plt.legend()

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


