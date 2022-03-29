import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fpa import FPA, GAU

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

class PAN_Unet(nn.Module):
    
    
    def __init__(self):
        super(PAN_Unet, self).__init__()

        self.dropout = nn.Dropout(0.25)
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

        # Bottleneck #
        self.convB1 = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.normB1 = nn.BatchNorm2d(512)
        #relu
        self.convB2 = nn.Conv2d(512,512,3, padding=1, bias=False)
        self.normB2 = nn.BatchNorm2d(512)

        # FPA #
        self.fpa = FPABlock(512,1024)
        
        
        #############

        #UP BLOCK 1#
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        self.norm9 = nn.BatchNorm2d(256)
 
        self.gau1 = GAU(512,256,upsample=True) # dont upsample the first one?
        #relu
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.norm10 = nn.BatchNorm2d(256)
        #relu
        
        
        #UP BLOCK 2#
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, 3, padding=1, bias=False)
        self.norm11 = nn.BatchNorm2d(128)

        self.gau2 = GAU(256,128,upsample=True)
        #relu
        self.conv12 = nn.Conv2d(128, 128, 3, padding=1, bias=False)
        self.norm12 = nn.BatchNorm2d(128)
        #relu
        
        #UP BLOCK 3#
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128,64,3,padding=1, bias=False)
        self.norm13 = nn.BatchNorm2d(64)

        self.gau3 = GAU(128,64,upsample=True)
        #relu
        self.conv14 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.norm14 = nn.BatchNorm2d(64)
        #relu
        
        #UP BLOCK 4#
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv15 = nn.Conv2d(64,32,3,padding=1, bias=False)
        self.norm15 = nn.BatchNorm2d(32)

        self.gau4 = GAU(64,32,upsample=True)
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

        # FEATURE PYRAMID ATTENTION
        x = self.fpa(x)
        
        #### DECODER ####
        
        #BLOCK 1
        # Use GAU module before next convolution
        """
        x = self.upconv1(x)
        x = torch.cat((x, enc4), dim=1) # skip layer
        x = self.conv9(x)
        x = F.relu(self.norm9(x))
        """
        
        x = self.gau1(x,enc4) # need to add this one up with the output of the high-level feature
        x = self.conv10(x)
        x = F.relu(self.norm10(x))
        
        #BLOCK 2
        """
        x = self.upconv2(x)
        x = torch.cat((x, enc3), dim=1)
        x = self.conv11(x)
        x = F.relu(self.norm11(x))
        """

        x = self.gau2(x,enc3)
        x = self.conv12(x)
        x = F.relu(self.norm12(x))
        
        #BLOCK 3
        """
        x = self.upconv3(x)
        x = torch.cat((x, enc2), dim=1)
        x = self.conv13(x)
        x = F.relu(self.norm13(x))
        """
        
        
        x = self.gau3(x,enc2)
        x = self.conv14(x)
        x = F.relu(self.norm14(x))
        
        #BLOCK 4
        """
        x = self.upconv4(x)
        x = torch.cat((x, enc1), dim=1)
        x = self.conv15(x) 
        x = F.relu(self.norm15(x))
        """

        x = self.gau4(x,enc1)
        x = self.conv16(x)
        x = F.relu(self.norm16(x))
        
        return self.convEND(x)


if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = PAN_Unet()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)