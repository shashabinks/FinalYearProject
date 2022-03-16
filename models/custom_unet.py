import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .fpa import FPA, GAU

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1

class RPAN_Unet(nn.Module):
    
    
    def __init__(self):
        super(RPAN_Unet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.dropout = nn.Dropout(0.25)

        self.RRCNN1 = RRCNN_block(ch_in=5,ch_out=32,t=2)

        self.RRCNN2 = RRCNN_block(ch_in=32,ch_out=64,t=2)
        
        self.RRCNN3 = RRCNN_block(ch_in=64,ch_out=128,t=2)
        
        self.RRCNN4 = RRCNN_block(ch_in=128,ch_out=256,t=2)

        # Bottleneck Layer #

        
        
        self.RRCNN5 = RRCNN_block(ch_in=256,ch_out=512,t=2)

        # FPA #
        self.fpa = FPA(channels=512)
        
        
        
        #############

        #UP BLOCK 1#
        self.gau1 = GAU(512,256) # dont upsample the first one?
        #relu
        self.Up_RRCNN4 = RRCNN_block(ch_in=256, ch_out=256,t=2)
        #relu
        
        
        #UP BLOCK 2#
        self.gau2 = GAU(256,128)
        #relu
        self.Up_RRCNN3 = RRCNN_block(ch_in=128, ch_out=128,t=2)
        #relu
        
        #UP BLOCK 3#
        self.gau3 = GAU(128,64)
        #relu
        self.Up_RRCNN2 = RRCNN_block(ch_in=64, ch_out=64,t=2)
        #relu
        

        #UP BLOCK 4#
        self.gau4 = GAU(64,32)
        #relu
        self.Up_RRCNN1 = RRCNN_block(ch_in=32, ch_out=32,t=2)
        #relu
        
        #1x1 Conv
        self.convEND = nn.Conv2d(32, 1, 1)
        
        
        
        
        
        
    def forward(self, x):
        x1 = self.RRCNN1(x)

        x2 = self.maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.maxpool(x3)
        x4 = self.RRCNN4(x4)

        # bottleneck Level 5
        x5 = self.maxpool(x4)
        x5 = self.RRCNN5(x5)
        x5 = self.fpa(x5)
        
        
    
        # decoding + concat path
        
        # Level 4
        d5 = self.gau1(x5,x4)
        d5 = self.Up_RRCNN4(d5)

        # Level 3
        d4 = self.gau2(d5, x3)
        d4 = self.Up_RRCNN3(d4)

        # Level 2
        d3 = self.gau3(d4, x2)
        d3 = self.Up_RRCNN2(d3)

        # Level 1
        d2 = self.gau4(d3,x1)
        d2 = self.Up_RRCNN1(d2)

        d1 = self.convEND(d2)

        return d1


if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = RPAN_Unet()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)