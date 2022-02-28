# A Dense Multi-Modal U-Net
from turtle import forward
import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def croppCenter(tensorToCrop,finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2,dtype=int)
    croppBorders[0] = int(diff[0]/2)
    croppBorders[1] = int(diff[1]/2)

    return tensorToCrop[:, :,
                        croppBorders[0]:croppBorders[0] + finalShape[2],
                        croppBorders[1]:croppBorders[1] + finalShape[3]]

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
            super(ConvBlock2d, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            
        )

    def forward(self, x):
        x = self.conv1(x)
        return x 

class UpBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock2d, self).__init__()
        self.up_conv = ConvTrans2d(in_ch, out_ch)
        self.conv = ConvBlock2d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


class DMM_Unet_4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_of_features = 32):
        super().__init__()

        self.in_dim = in_channels
        self.out_dim = num_of_features
        self.final_out_dim = out_channels


        # ~~~ ENCODING PATHS ~~~ #

        # 5 image modalities, therefore 5 separate paths, all the same structure

        # Encoder ~ M1 ~ CT
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_0 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_4_0 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M2 ~ CT_CBV
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_1 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_1 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M3 ~ CT_CBF
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_2 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_2 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M4 ~ CT_Tmax
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_3 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_3 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        

        # ~~~ BRIDGE ~~~ #
        self.bridge = ConvBlock2d(self.out_dim * 60, self.out_dim * 16 )

        # ~~~ DECODER PATH ~~~ #
        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)


        # ~~~ OUTPUT ~~~ #
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
    

    def forward(self, input):

        # define all 5 image modalities
        # each image is defined as: batch_sze x modal x height x width
        m1 = input[:,0:1,:,:]
        m2 = input[:,1:2,:,:]
        m3 = input[:,2:3,:,:]
        m4 = input[:,3:4,:,:]
        


        # ~~~ ENCODING ~~~ #
        # ~~~ L1 ~~~ #
        down_1_0 = self.down_1_0(m1)  # Batch Size * outdim * volume_size * height * width
        down_1_1 = self.down_1_1(m2)
        down_1_2 = self.down_1_2(m3)
        down_1_3 = self.down_1_3(m4)
        


        # ~~~ L2 ~~~ #
        
        # prepare the input
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0)), dim=1)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2)), dim=1)
        
        

        #print(input_2nd_0.shape)
        # do the convolution
        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)
        

        
        # ~~~ L3 ~~~ #
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        down_2_2m = self.pool_2_0(down_2_2)
        down_2_3m = self.pool_2_0(down_2_3)
        

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m,down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m,down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m,down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)
        

        # ~~~ L4 ~~~ #
        down_3_0m = self.pool_2_0(down_3_0)
        down_3_1m = self.pool_2_0(down_3_1)
        down_3_2m = self.pool_2_0(down_3_2)
        down_3_3m = self.pool_2_0(down_3_3)
        

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        

        down_4_0 = self.down_4_0(input_4th_0)
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)
        

        
        # ~~~ BRIDGE ~~~ #
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)
        

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)

        bridge = self.bridge(inputBridge)

        # ~~~~~~ Decoding path ~~~~~~~  #
        skip_1 = (down_4_0 + down_4_1 + down_4_2 + down_4_3) / 4.0  # most bottom one
        skip_2 = (down_3_0 + down_3_1 + down_3_2 + down_3_3) / 4.0
        skip_3 = (down_2_0 + down_2_1 + down_2_2 + down_2_3) / 4.0
        skip_4 = (down_1_0 + down_1_1 + down_1_2 + down_1_3) / 4.0      # top one


        x = self.upLayer1(bridge, skip_1)
        x = self.upLayer2(x, skip_2)
        x = self.upLayer3(x, skip_3)
        x = self.upLayer4(x, skip_4)

        return self.out(x)

if __name__ == "__main__":
    batch_size = 4
    num_classes = 1  # one hot
    initial_kernels = 32

    net = DMM_Unet_4(1, num_classes)
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 4, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)