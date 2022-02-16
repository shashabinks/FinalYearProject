### this is a 2D dataset loader ###
from configparser import Interpolation
from matplotlib import transforms
from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.tensor as ts
import torchvision
import torch.nn as nn
import torchio as tio
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import SimpleITK as sitk
import random

# 5-Channel Loader
class ISLES2018_loader(Dataset):
    def __init__(self, file_dir, modalities=None):
        super().__init__()
        
        self.samples = []

        for case_name in os.listdir(file_dir):
            case_path = os.path.join(file_dir, case_name)
            case = {} # dict which will store all the images for each modality, this is later used to iterate through all the slices etc

            for path in os.listdir(case_path):
                modality = re.search(r'SMIR.Brain.XX.O.(\w+).\d+',path).group(1)
                if modality != 'CT_4DPWI': # leave DWI images out for now, can be trained with later
                    nii_path_name = os.path.join(case_path,path,path+'.nii')
                    img = nib.load(nii_path_name)
                    case[modality] = img
            
            normalize = torchvision.transforms.Normalize([0.5],[0.5])

            for i in range(case['CT'].shape[2]):        # go through each case dimension (2-22)
                slices = []                                # create array for each image slice
                for modality in modalities:             # loop through the modalities
                    if modality != 'OT':                # ignore the ground truth

                        # add image augmentations here 

                        slice = case[modality].get_fdata()[:,:,i]
                        
                        #slice = self.n4BiasCorrection(slice)
                        
                        image_slice = torch.from_numpy(slice).float().unsqueeze(0) # image slice converted to torch tensor
                        
                        # normalize image
                        image_slice = normalize(image_slice)

                        slices.append(image_slice) # add the slice to the array
                
                gt_slice = torch.from_numpy(case['OT'].get_fdata()[:,:,i]).float().unsqueeze(0) # slice of the corresponding ground_truths
                
                gt_slice = normalize(gt_slice)

                # now transform all the slices in the array before concatenating them:
                # slices, gt_slice = self.transform(slices,gt_slice)

                combined = torch.cat(tuple(slices), dim=0) # concatenate all the slices to form 5 channel, input has to be a set
                
                  
                self.samples.append((combined, gt_slice))  # append tuples of combined slices and ground truth masks, this makes it easier to later compare the pred/actual
        
            
        
                            
    
    def __getitem__(self, idx):
        return self.samples[idx]   # return the dataset corresponding to the input modality
    
    def __len__(self):      # return length of dataset for each modality
        return len(self.samples)

    def transform(self, slices, gt):
        
        # flip horizontally randomly
        if random.random() > 0.5:
            for i in range(len(slices)):
                image = slices[i]
                image = TF.hflip(image)
                slices[i] = image
            
            gt = TF.hflip(image)

        
        return slices, gt

    # here we want to apply different augmentations 
    def normalizes(self,img):
        # apply bias correction
        # rescale image
        #rescale = tio.Resize() 128 x 128 x 32

        # convert img to tensor
        #img = img.get_fdata()
        
        # need this to add feature channel
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        t_img = torch.from_numpy(img)

        # normalize image
        normalize = tio.ZNormalization()
        t_img = normalize(t_img)

    
        return t_img
    
    # this bit is a huge bottleneck
    def n4BiasCorrection(self, img):
        
        img = sitk.GetImageFromArray(img)
        image = sitk.Cast(img, sitk.sitkFloat32)
        corrected_img = sitk.N4BiasFieldCorrection(image)
        img = sitk.GetArrayFromImage(corrected_img)

        return img
    
    # returns the modality and ground truth image
    def getData(self,modality):
        return self.data[modality], self.data['OT']

            
            

directory = "ISLES/TRAINING"
modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset = ISLES2018_loader(directory, modalities)
#print(dataset.__len__())



