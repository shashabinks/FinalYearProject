from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# take all the 3D images, and sort them under each modality
# i.e. {CT:[ ], CBV:[], ...}

class ISLES2018_loader(Dataset):
    def __init__(self, folder, modalities=None):
        super().__init__()
        self.modalities = modalities
        #self.samples = []
        #self.cases = {}

        self.data = {}

        for modality in modalities:
            self.data.update({modality : []})

        
        for case_name in os.listdir(folder):
            case_path = os.path.join(folder, case_name)
            

            
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

    def view_slices(self):
        for image in self.data["CT_CBV"]:
            img = image.get_fdata()
            print(img.shape)
            #plt.imshow(section,cmap='gray', alpha=1)

            #plt.colorbar(label='intensity')
            #plt.show()
            







directory = "ISLES/TRAINING"
modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
dataset = ISLES2018_loader(directory, modalities)
#dataset.viewData()
dataset.view_slices()

