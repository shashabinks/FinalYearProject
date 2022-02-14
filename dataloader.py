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
import torch.nn as nn
import torchio as tio
import torch.nn.functional as F
import torch.optim as optim
from mpl_toolkits.mplot3d import Axes3D
import monai

# take all the 3D images, and sort them under each modality
# i.e. {CT:[ ], CBV:[], ...}
# this class will return all the images needed by the loader
class ISLES2018_loader(Dataset):
    def __init__(self, folder, modalities=None):
        super().__init__()
        self.modalities = modalities
        self.data = {}

        for modality in modalities:
            self.data.update({modality : []})

        # this part just loads in the nib files, ideally we want these to be in the form of np arrays
        # iterate through all the case folders 
        for case_name in os.listdir(folder):
            case_path = os.path.join(folder, case_name)
            
            # for each file in the current case directory, find all the modalities and append to list
            for file_path in os.listdir(case_path):
                modality = re.search(r'SMIR.Brain.XX.O.(\w+).\d+',file_path).group(1)
                
                if modality != 'CT_4DPWI': # ignore dwi images for now
                    nii_path_name = os.path.join(case_path,file_path,file_path+'.nii')
                    img = nib.load(nii_path_name)
                    # maybe apply transformations here then append to the overall data file
                    self.data[modality].append(img)

            """
            for file_path in os.listdir(case_path):
                modality = re.search(r'SMIR.Brain.XX.O.(\w+).\d+',file_path).group(1)
                if modality != 'CT_4DPWI': 
                    nii_path_name = os.path.join(case_path,file_path,file_path+'.nii')
                    img = nib.load(nii_path_name)
                    case[modality] = img
            
            self.cases = case
            """
            

        ## TODO: implement conversion from .nii to tensor, maybe update the data variable
           
    
    def __getitem__(self, modality):
        return self.data[modality]   # return the dataset corresponding to the input modality
    
    def __len__(self):      # return length of dataset for each modality
        return len(self.data["CT"])

    def viewData(self):
        print(self.data)
    
    def normalise(self,arr):
        
        arr_min = np.min(arr)
        return (arr-arr_min)/(np.max(arr)-arr_min)
    
    def view_slices(self):
        for image in self.data["CT_CBV"]:
            img = image.get_fdata()
            #print(img.shape)
            img = img.reshape(1,256,256,8)
            #print(new.shape)

            #HOUNSFIELD_AIR, HOUNSFIELD_BONE = 10, 45 # cbv
            #HOUNSFIELD_AIR, HOUNSFIELD_BONE = -200, 90 # mtt

            x_tensor = torch.from_numpy(img)
            print(x_tensor.shape)

            #plt.imshow(img[:,:,1],cmap='gray', alpha=1)
            #plt.colorbar(label='intensity')
            
            #transforms = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
            transforms = tio.ZNormalization() # this is using the same formula as: img_modality = (img_modality - mean_modality) / std_modality 

            #ct_air, ct_bone = 10, 200
            #rescale = tio.RescaleIntensity(
                #out_min_max=(0, 1),percentiles=(0.5, 99.5), in_min_max=(0, 1))
            #ct_normalized = rescale(x_tensor)

            new_img = transforms(x_tensor)
            print(new_img.shape)

            plt.imshow(new_img[0,:,:,3].unsqueeze(0))
            plt.show()
            break
    
    # returns the modality and ground truth image
    def getData(self,modality):
        return self.data[modality], self.data['OT']

            
            

#directory = "ISLES/TRAINING"
#modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
#dataset = ISLES2018_loader(directory, modalities)
#dataset.viewData()
#dataset.view_slices()

