### this is a 2D dataset loader ###

from torch.utils.data import Dataset
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torchvision.transforms.functional as TF

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
                for i in range(case['CT'].shape[2]):        # go through each case dimension (2-22)
                    slices = []                                # create array for each image slice
                    for modality in modalities:             # loop through the modalities
                        if modality != 'OT':                # ignore the ground truth

                            slice=case[modality].get_fdata()
                            img_array = np.array(slice).astype('float64')
                            img_2d = img_array[:,:,i].transpose((1,0))
                            img_2d = np.uint8(img_2d[None,:])
                            img_2d = torch.from_numpy(img_2d)


                            slices.append(img_2d) # add the slice to the array
                    

                    gt_slice=case['OT'].get_fdata()
                    gt_array = np.array(gt_slice).astype('float64')
                    gt_2d = gt_array[:,:,i].transpose((1,0))
                    gt_2d = np.uint8(gt_2d[None,:])
                    gt_2d = torch.from_numpy(gt_2d)

                    


                    # augment original images to produce new data
                    slices, slices_h, slices_v, slices_r, gt_slice , gt_h, gt_v, gt_r = self.transform(slices,gt_2d)
                    
                    combined = torch.cat(tuple(slices), dim=0) # concatenate all the slices to form 5 channel, input has to be a set
                    combined_h = torch.cat(tuple(slices_h), dim=0)
                    combined_v = torch.cat(tuple(slices_v), dim=0)
                    combined_r = torch.cat(tuple(slices_r), dim=0)

                    self.samples.append((combined, gt_slice))
                    self.samples.append((combined_h, gt_h))
                    self.samples.append((combined_v, gt_v))
                    self.samples.append((combined_r, gt_r))  # append tuples of combined slices and ground truth masks, this makes it easier to later compare the pred/actual

                

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



        # copy the original images
        slices_1, gt_1 = slices.copy(), gt.copy()
        slices_2, gt_2 = slices.copy(), gt.copy()
        slices_3, gt_3 = slices.copy(), gt.copy()


        if augment:
            # flip horizontally randomly the original images
            if random.random() >= 0.0:
                for i in range(len(slices)):
                    image = slices_1[i]
                    image = TF.hflip(image)
                    slices_1[i] = image
                
                gt_1 = TF.hflip(gt)
            
        
            # flip vertically randomly the original images
            if random.random() >= 0.0:
                for i in range(len(slices)):
                    image = slices_2[i]
                    image = TF.vflip(image)
                    slices_2[i] = image
                
                gt_2 = TF.vflip(gt)
            
            # rotate at random the original images
            if random.random() >= 0.0:
                angle = random.randint(-30, 30)
                for i in range(len(slices)):
                    image = slices_3[i]
                    image = TF.rotate(image, angle, fill=(0,))
                    slices_3[i] = image
                
                gt_3 = TF.rotate(gt, angle, fill=(0,))
        
        

        # convert back to tensor/normalize
        for i in range(len(slices)):

            
            image1 = slices[i]
            image1 = TF.to_tensor(image1)
            slices[i] = image1

        gt = TF.to_tensor(gt)

        for i in range(len(slices)):

            
            image2 = slices_1[i]
            image2 = TF.to_tensor(image2)
            slices_1[i] = image2
            
        gt_1 = TF.to_tensor(gt_1)

        for i in range(len(slices)):

            
            image3 = slices_2[i]
            image3 = TF.to_tensor(image3)
            slices_2[i] = image3
            
        gt_2 = TF.to_tensor(gt_2)

        for i in range(len(slices)):

            image4 = slices_3[i]
            image4 = TF.to_tensor(image4)
            slices_3[i] = image4
        
        gt_3 = TF.to_tensor(gt_3)

    
        return slices, slices_1, slices_2, slices_3, gt, gt_1 , gt_2, gt_3
      

    
    # returns the modality and ground truth image
    def getData(self,modality):
        return self.data[modality], self.data['OT']


class val_ISLES2018_loader(Dataset):
    def __init__(self, dataset,modalities=None):
        super().__init__()
        
        self.samples = []
        
        for patient in dataset:
            for case in patient:
                for i in range(case['CT'].shape[2]):        # go through each case dimension (2-22)
                    slices = []                                # create array for each image slice
                    for modality in modalities:             # loop through the modalities
                        if modality != 'OT':                # ignore the ground truth

                            slice=case[modality].get_fdata()
                            img_array = np.array(slice).astype('float64')
                            img_2d = img_array[:,:,i].transpose((1,0))
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
            if modality != 'CT_4DPWI': # leave DWI images out for now, can be trained with later
                nii_path_name = os.path.join(case_path,path,path+'.nii')
                img = nib.load(nii_path_name)
                case[modality] = img
        
        case_num += 1 # increase the case num
        dataset[patient_id-1].append(case)
    
    return dataset

    

    

        
#directory = "ISLES/TRAINING"
#modalities = ['OT', 'CT', 'CT_CBV', 'CT_CBF', 'CT_Tmax' , 'CT_MTT']
#dataset = load_data(directory)
#print(dataset.__len__())



