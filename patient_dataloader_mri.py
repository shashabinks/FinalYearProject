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
import albumentations as A
import random
import lookup


# TODO: Add a much simpler way to split the dataset, i.e. store all the images in a dictionary of lists' and 
# split the list accordingly using only simple list partioning (maybe look into other ways later)
# 


# 5-Channel Loader
class train_ISLES2018_loader(Dataset):
    def __init__(self, dataset,modalities=None):
        super().__init__()
        
        self.samples = []
        
        for patient in dataset:
            for case in patient:  
                    
                

                for t in range(0,1):
                    for i in range(case['CT_4DPWI'].shape[2]):
                        slices = []
                        slice=case['CT_4DPWI'].get_fdata()
                        img_array = np.array(slice).astype('float64')
                        img_2d = img_array[:,:,i,t].transpose((1,0))
                        
                        
                        img_2d = np.uint8(img_2d[None,:])
                        img_2d = torch.from_numpy(img_2d)


                        slices.append(img_2d) # add the slice to the array
                        

                        gt_slice=case['OT'].get_fdata()
                        gt_array = np.array(gt_slice).astype('float64')
                        gt_2d = gt_array[:,:,i].transpose((1,0))
                        gt_2d = np.uint8(gt_2d[None,:])
                        gt_2d = torch.from_numpy(gt_2d)

                        
                    
                        slices, gt_slice = self.transform(slices,gt_2d)
                        
                        combined = torch.cat(tuple(slices), dim=0) # concatenate all the slices to form 5 channel, input has to be a set
                        
                    
                        self.samples.append((combined, gt_slice))  # append tuples of combined slices and ground truth masks, this makes it easier to later compare the pred/actual
                      
    def __getitem__(self, idx):
        return self.samples[idx]   # return the dataset corresponding to the input modality
    
    def __len__(self):      # return length of dataset for each modality
        return len(self.samples)

    def transform(self, slices, gt):

        augment = True

        # convert each slice into a pil image
        for i in range(len(slices)):
            image = slices[i]
            image = TF.to_pil_image(image)
            slices[i] = image
        
        gt = TF.to_pil_image(gt)

       



        # resize slices
        for i in range(len(slices)):
            image = slices[i]
            image = image.resize((256, 256))
            slices[i] = image
        
        gt =  gt.resize((256, 256))

        if augment:
            # flip horizontally randomly
            if random.random() > 0.5:
                for i in range(len(slices)):
                    image = slices[i]
                    image = TF.hflip(image)
                    slices[i] = image
                
                gt = TF.hflip(gt)
            
            # flip vertically randomly
            if random.random() > 0.5:
                for i in range(len(slices)):
                    image = slices[i]
                    image = TF.vflip(image)
                    slices[i] = image
                
                gt = TF.vflip(gt)
            
            # rotate at random
            if random.random() > 0.5:
                angle = random.randint(-30, 30)
                for i in range(len(slices)):
                    image = slices[i]
                    image = TF.rotate(image, angle, fill=(0,))
                    slices[i] = image
                
                gt = TF.rotate(gt, angle, fill=(0,))

        # convert back to tensor/normalize
        for i in range(len(slices)):
            image = slices[i]
            image = TF.to_tensor(image)
            slices[i] = image
        
        gt = TF.to_tensor(gt)

    
        return slices, gt
      
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


class val_ISLES2018_loader(Dataset):
    def __init__(self, dataset,modalities=None):
        super().__init__()
        
        self.samples = []
        
        for patient in dataset:
            for case in patient:  
                    
                for t in range(0,1):
                    for i in range(case['CT_4DPWI'].shape[2]):
                        slices = []
                        slice=case['CT_4DPWI'].get_fdata()
                        img_array = np.array(slice).astype('float64')
                        img_2d = img_array[:,:,i,t].transpose((1,0))
                        
                        
                        img_2d = np.uint8(img_2d[None,:])
                        img_2d = torch.from_numpy(img_2d)


                        slices.append(img_2d) # add the slice to the array
                        

                        gt_slice=case['OT'].get_fdata()
                        gt_array = np.array(gt_slice).astype('float64')
                        gt_2d = gt_array[:,:,i].transpose((1,0))
                        gt_2d = np.uint8(gt_2d[None,:])
                        gt_2d = torch.from_numpy(gt_2d)

                        
                    
                        slices, gt_slice = self.transform(slices,gt_2d)
                        
                        combined = torch.cat(tuple(slices), dim=0) # concatenate all the slices to form 5 channel, input has to be a set
                        
                    
                        self.samples.append((combined, gt_slice))   # append tuples of combined slices and ground truth masks, this makes it easier to later compare the pred/actual
               
                        
    def __getitem__(self, idx):
        return self.samples[idx]   # return the dataset corresponding to the input modality
    
    def __len__(self):      # return length of dataset for each modality
        return len(self.samples)

    def transform(self, slices, gt):
        # convert each slice into a pil image
        for i in range(len(slices)):
            image = slices[i]
            image = TF.to_pil_image(image)
            slices[i] = image
        
        gt = TF.to_pil_image(gt)

        
        # resize slices
        for i in range(len(slices)):
            image = slices[i]
            image = image.resize((256, 256))
            slices[i] = image
        
        gt =  gt.resize((256, 256))
        
        # convert back to tensor/normalize
        for i in range(len(slices)):
            image = slices[i]
            image = TF.to_tensor(image)
            slices[i] = image
        
        gt = TF.to_tensor(gt)

        
        return slices, gt
    

def load_data(file_dir):
    dataset = [[] for _ in range(0,63)] 
    case_num = 1
    for case_name in os.listdir(file_dir):

        # some how get the case number and compare that to patient id lookup table
        # then, for each case, we append the images to a list and append the resulting case list to a list under the patient id in a dictionary   
        # so the result is patients = {patient_1 : [[case_1_images], [case_2_images]], etc}  
        patient_id = lookup.patients_train[case_num]

        case_path = os.path.join(file_dir, case_name)
        case = {} # dict which will store all the images for each modality, this is later used to iterate through all the slices etc

        for path in os.listdir(case_path):
            modality = re.search(r'SMIR.Brain.XX.O.(\w+).\d+',path).group(1)
            if modality == 'CT_4DPWI' or modality == 'OT': # leave DWI images out for now, can be trained with later
                nii_path_name = os.path.join(case_path,path,path+'.nii')
                img = nib.load(nii_path_name)
                case[modality] = img

                if modality == 'CT_4DPWI':
                    print(img.shape)
        
        case_num += 1 # increase the case num
        dataset[patient_id-1].append(case)
    
    print(dataset[1])
    
    return dataset

    

    

        
#directory = "ISLES/TRAINING"
#modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
#dataset = load_data(directory)
#print(dataset.__len__())



