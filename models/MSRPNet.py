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
import math
from typing import Tuple, Dict

"""
Implementation of Pyramid Attention Network (Modules Only)

Li, H., Xiong, P., An, J. and Wang, L., 2018. Pyramid attention network for semantic segmentation. arXiv preprint arXiv:1805.10180.
"""
class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        add_relu: bool = True,
        interpolate: bool = False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
        self.add_relu = add_relu
        self.interpolate = interpolate
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        if self.interpolate:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x




class FPABlock(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_mode="bilinear"):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == "bilinear":
            self.align_corners = True
        else:
            self.align_corners = False

        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=7,
                stride=1,
                padding=3,
            ),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1),
        )
        self.conv2 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.conv1 = ConvBnRelu(in_channels=1, out_channels=1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        upscale_parameters = dict(mode=self.upscale_mode, align_corners=self.align_corners)
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)

        mid = self.mid(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)

        x = torch.mul(x, mid)
        x = x + b1
        return x

class Conv2d_batchnorm(nn.Module):
  def __init__(self,input_features : int,num_of_filters : int ,kernel_size : Tuple = (2,2),stride : Tuple = (1,1), activation : str = 'relu',padding  : int= 0)->None:
    """
    Arguments:
      x - input layer
      num_of_filters - no. of filter outputs
      filters - shape of the filters to be used
      stride - stride dimension 
      activation -activation function to be used
    Returns - None
    """
    super().__init__()
    self.activation = activation
    self.conv1 = nn.Conv2d(in_channels=input_features,out_channels=num_of_filters,kernel_size=kernel_size,stride=stride,padding = padding)
    self.batchnorm = nn.BatchNorm2d(num_of_filters,affine=False)
  
  def forward(self,x : torch.Tensor)->torch.Tensor:
    x = self.conv1(x)
    x = self.batchnorm(x)
    if self.activation == 'relu':
      return F.relu(x)
    else:
      return x

class Multiresblock(nn.Module):
  def __init__(self,input_features : int, corresponding_unet_filters : int ,alpha : float =1.67)->None:
    """
        MultiResblock
        Arguments:
          x - input layer
          corresponding_unet_filters - Unet filters for the same stage
          alpha - 1.67 - factor used in the paper to dervie number of filters for multiresunet filters from Unet filters
        Returns - None
    """ 
    super().__init__()
    self.corresponding_unet_filters = corresponding_unet_filters
    self.alpha = alpha
    self.W = corresponding_unet_filters * alpha
    self.conv2d_bn_1x1 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167)+int(self.W*0.333)+int(self.W*0.5),
    kernel_size = (1,1),activation='None',padding = 0)

    self.conv2d_bn_3x3 = Conv2d_batchnorm(input_features=input_features,num_of_filters = int(self.W*0.167),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.conv2d_bn_5x5 = Conv2d_batchnorm(input_features=int(self.W*0.167),num_of_filters = int(self.W*0.333),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.conv2d_bn_7x7 = Conv2d_batchnorm(input_features=int(self.W*0.333),num_of_filters = int(self.W*0.5),
    kernel_size = (3,3),activation='relu',padding = 1)
    self.batch_norm1 = nn.BatchNorm2d(int(self.W*0.5)+int(self.W*0.167)+int(self.W*0.333) ,affine=False)

  def forward(self,x: torch.Tensor)->torch.Tensor:

    temp = self.conv2d_bn_1x1(x)
    a = self.conv2d_bn_3x3(x)
    b = self.conv2d_bn_5x5(a)
    c = self.conv2d_bn_7x7(b)
    x = torch.cat([a,b,c],axis=1)
    x = self.batch_norm1(x)
    x += temp
    x = self.batch_norm1(x)
    return x

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
                                   , nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
                                   , nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
    
    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        
        return x

class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UpSample,self).__init__()

        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2))
        

    def forward(self,x):

        x = self.conv1(x)

        return x

class conv2d_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size,activation=True):
        super(conv2d_block, self).__init__()

        if activation:
          self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=1, bias=False),
              nn.BatchNorm2d(out_ch),
              nn.ReLU(inplace=True)
          )
        else:
          self.conv = nn.Sequential(
              nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, bias=True),
              nn.BatchNorm2d(out_ch)
          )
          
    def forward(self, x):
        x = self.conv(x)
        return x



def output1(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.Conv2d(32, n_classes, kernel_size=1)
    )


def output2(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.Conv2d(32, n_classes, kernel_size=1)
    )


def output3(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.Conv2d(32, n_classes, kernel_size=1)
    )






class UNet_2D(nn.Module):
    
    
    def __init__(self, in_channels=5,fpa_block=False,sa=False,deep_supervision=False, mhca=False, respath=False):
        super(UNet_2D, self).__init__()

        self.fpa_block = fpa_block
        self.sa = sa
        self.deep_supervision = deep_supervision
        self.mhca = mhca
        self.respath = respath

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        
        self.fpa = FPABlock(256,256)



        # down path
        self.doubleconv1= DoubleConv(in_channels,32)
        self.doubleconv2= DoubleConv(32,64)
        self.doubleconv3= DoubleConv(64,128)
        self.doubleconv4= DoubleConv(128,256)

        # bottleneck 

        self.doubleconv5= DoubleConv(256,512)

        # up path
        self.up1 = UpSample(512,256)
        self.doubleconv6 = DoubleConv(512,256)

        self.up2 = UpSample(256,128)
        self.doubleconv7 = DoubleConv(256,128)

        self.up3 = UpSample(128,64)
        self.doubleconv8 = DoubleConv(128,64)

        self.up4 = UpSample(64,32)
        self.doubleconv9 = DoubleConv(64,32)

        self.convEND = nn.Conv2d(32, 1, 1)

        self.output1 = output1(1)
        self.output2 = output2(1)
        self.output3 = output3(1)
        
        
        
        
        
        
    def forward(self, x):
        #### ENCODER ####
        
        #BLOCK 1
        x = self.doubleconv1(x)
        skip_1 = x
        x = self.pool(x)
        
        #BLOCK 2
        x = self.doubleconv2(x)
        skip_2 = x
        x = self.pool(x)
        
        #BLOCK 3
        x = self.doubleconv3(x)
        skip_3 = x
        x = self.pool(x)
        
        #BLOCK 4
        x = self.doubleconv4(x)
        skip_4 = x
        

        #BOTTLENECK
        # enable feature pyramid attention module
        if self.fpa_block:
            x = self.fpa(x)

        else:
            # original bottleneck
            x = self.pool(x)
            x = self.doubleconv5(x)
            x = self.up1(x)
        
        #### DECODER ####
        
        #BLOCK 1

        x = torch.cat((x, skip_4), dim=1) # skip layer
        x = self.doubleconv6(x)

        if self.training and self.deep_supervision:
            x1 = self.output1(x)
        
        #BLOCK 2
        x = self.up2(x)
        x = torch.cat((x, skip_3), dim=1) # skip layer
        x = self.doubleconv7(x)

        if self.training and self.deep_supervision:
            x2 = self.output2(x)


        #BLOCK 3
        x = self.up3(x)
        x = torch.cat((x, skip_2), dim=1) # skip layer
        x = self.doubleconv8(x)

        if self.training and self.deep_supervision:
            x3 = self.output3(x)

        
        #BLOCK 4


        x = self.up4(x)
        x = torch.cat((x, skip_1), dim=1) # skip layer
        x = self.doubleconv9(x)

        if self.training and self.deep_supervision:
            return self.convEND(x), x1 , x2, x3
        
        return self.convEND(x)

if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = UNet_2D()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)