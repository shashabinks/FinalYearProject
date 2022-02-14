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
                
                if modality != 'CT_4DPWI': 
                    nii_path_name = os.path.join(case_path,file_path,file_path+'.nii')
                    img = nib.load(nii_path_name)
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


            x_tensor = torch.from_numpy(img)
            print(x_tensor.shape)

            #plt.imshow(img[:,:,1],cmap='gray', alpha=1)
            #plt.colorbar(label='intensity')
            
            transforms = tio.CropOrPad((120,120,32))
            new_img = transforms(x_tensor)
            print(new_img.shape)
            #plt.show()
            break 
            
            

directory = "ISLES/TRAINING"
modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset = ISLES2018_loader(directory, modalities)
#dataset.viewData()
dataset.view_slices()

