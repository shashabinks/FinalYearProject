import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP,self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)
 
    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = nn.functional.interpolate(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class UNet_ASPP(nn.Module):
    
    
    def __init__(self):
        super(UNet_ASPP, self).__init__()
        # DOWN BLOCK 1 #
        self.conv1 = nn.Conv2d(5, 32, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)
        #relu
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(32)
        #Relu
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #DOWN BLOCK 2 #
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(64)
        #relu
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(64)
        #relu
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #DOWN BLOCK 3 #
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.norm5 = nn.BatchNorm2d(128)
        #relu
        self.conv6 = nn.Conv2d(128,128, 3, padding=1, bias=False)
        self.norm6 = nn.BatchNorm2d(128)
        #relu
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #DOWN BLOCK 4 #
        self.conv7 = nn.Conv2d(128,256, 3, padding=1, bias=False)
        self.norm7 = nn.BatchNorm2d(256)
        #relu
        self.conv8 = nn.Conv2d(256,256, 3, padding=1, bias=False)
        self.norm8 = nn.BatchNorm2d(256)
        #relu
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        #############
        
        #Bottleneck #
        self.convB1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.normB1 = nn.BatchNorm2d(512)
        #relu
        self.convB2 = nn.Conv2d(512,512,3, padding=1, bias=False)
        self.normB2 = nn.BatchNorm2d(512)

        self.aspp = ASPP(512,512)
        
        
        #############
        
        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.norm9 = nn.BatchNorm2d(256)
        #relu
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.norm10 = nn.BatchNorm2d(256)
        #relu
        
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.norm11 = nn.BatchNorm2d(128)
        #relu
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm12 = nn.BatchNorm2d(128)
        #relu
        
        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128,64,3,padding=1, bias=False)
        self.norm13 = nn.BatchNorm2d(64)
        #relu
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.norm14 = nn.BatchNorm2d(64)
        #relu
        
        #UP BLOCK 4#
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(64,32,3,padding=1, bias=False)
        self.norm15 = nn.BatchNorm2d(32)
        #relu
        self.conv16 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
        self.norm16 = nn.BatchNorm2d(32)
        #relu
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)
        
        
        
        
        
        
    def forward(self, x):
        #### ENCODER ####
        
        #BLOCK 1
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.conv2(x)
        enc1 = F.relu(self.norm2(x))
        x = self.pool1(enc1)
        
        #BLOCK 2
        x = self.conv3(x)
        x = F.relu(self.norm3(x))
        x = self.conv4(x)
        enc2 = F.relu(self.norm4(x))
        x = self.pool2(enc2)
        
        #BLOCK 3
        x = self.conv5(x)
        x = F.relu(self.norm5(x))
        x = self.conv6(x)
        enc3 = F.relu(self.norm6(x))
        x = self.pool3(enc3)
        
        #BLOCK 4
        x = self.conv7(x)
        x = F.relu(self.norm7(x))
        x = self.conv8(x)
        enc4 = F.relu(self.norm8(x))
        x = self.pool4(enc4)

        
        #BOTTLENECK
        x = self.convB1(x)
        x = F.relu(self.normB1(x))
        x = self.convB2(x)
        x = F.relu(self.normB2(x))
        
        x = self.aspp(x)
        
        #### DECODER ####
        
        #BLOCK 1
        x = self.upconv1(x)
        x = torch.cat((x, enc4), dim=1) # skip layer
        x = self.conv9(x)
        x = F.relu(self.norm9(x))
        x = self.conv10(x)
        x = F.relu(self.norm10(x))
        
        #BLOCK 2
        x = self.upconv2(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.conv11(x)
        x = F.relu(self.norm11(x))
        x = self.conv12(x)
        x = F.relu(self.norm12(x))
        
        #BLOCK 3
        x = self.upconv3(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.conv13(x)
        x = F.relu(self.norm13(x))
        x = self.conv14(x)
        x = F.relu(self.norm14(x))
        
        #BLOCK 4
        x = self.upconv4(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.conv15(x)
        x = F.relu(self.norm15(x))
        x = self.conv16(x)
        x = F.relu(self.norm16(x))
        
        return self.convEND(x)

if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = UNet_ASPP()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)