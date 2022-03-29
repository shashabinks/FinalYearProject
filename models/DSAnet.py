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

"""
Implementation of Pyramid Attention Network (Modules Only)

Li, H., Xiong, P., An, J. and Wang, L., 2018. Pyramid attention network for semantic segmentation. arXiv preprint arXiv:1805.10180.
"""
class FPA(nn.Module):
    def __init__(self, channels=512):
        """
        Feature Pyramid Attention
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels/4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels_mid)

        # Convolution Upsample
        self.conv_upsample_3 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_3 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_2 = nn.ConvTranspose2d(channels_mid, channels_mid, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_2 = nn.BatchNorm2d(channels_mid)

        self.conv_upsample_1 = nn.ConvTranspose2d(channels_mid, channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_upsample_1 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        :param x: Shape: [b, c, h, w]
        :return: out: Feature maps. Shape: [b, c, h, w]
        """
        # Master branch
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)

        # Merge branch 1 and 2
        x3_upsample = self.relu(self.bn_upsample_3(self.conv_upsample_3(x3_2)))
        x2_merge = self.relu(x2_2 + x3_upsample)
        x2_upsample = self.relu(self.bn_upsample_2(self.conv_upsample_2(x2_merge)))
        x1_merge = self.relu(x1_2 + x2_upsample)

        x_master = x_master * self.relu(self.bn_upsample_1(self.conv_upsample_1(x1_merge)))

        out = self.relu(x_master + x_gpb)

        return out

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

def positional_encoding_2d(d_model, height, width, device):
    """
    reference: wzlxjtu/PositionalEncoding2D
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                        "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width, device=device)

    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2, device=device) *
                        -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width, device=device).unsqueeze(1)
    pos_h = torch.arange(0., height, device=device).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe

def attention(q, k, v, d_k, mask=None, dropout=None):   
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.0):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        
    
        return self.out(concat)
    
class MHSABlock(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.0, pos_enc = True):
        super().__init__()
        self.attention = MultiHeadAttention(heads, d_model, dropout)
        self.pos_enc = pos_enc
        
    def forward(self, x):
        b, c, h, w = x.size()

        x2 = x        
        if self.pos_enc:
            pe = positional_encoding_2d(c, h, w, x.device)
            x2 = x2 + pe

        x2 = x2.reshape(b, c, h*w).permute(0, 2, 1)

        att = self.attention(x2, x2, x2).permute(0, 2, 1).reshape(b, c, h, w)
        
        return att

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

class ResPath(nn.Module):
    def __init__(self, in_ch, out_ch, length):
      super(ResPath,self).__init__()
      self.len = length
      self.conv3layers = nn.ModuleList([conv2d_block(in_ch,out_ch,3) for i in range(length)])
      self.conv1layers = nn.ModuleList([conv2d_block(in_ch,out_ch,1,activation=False) for i in range(length)])
      self.activation = nn.ReLU(inplace=True)
      self.Batch = nn.ModuleList([nn.BatchNorm2d(out_ch) for i in range(length)])
    
    def forward(self,x):
      out = x
      for i in range(self.len):
        shortcut = out
        shortcut = self.conv1layers[i](shortcut)
        out = self.conv3layers[i](out)
        out = torch.add(shortcut,out)
        out = self.activation(out)
        out = self.Batch[i](out)
      
      return out

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


class MHCABlock(nn.Module):
    def __init__(self, heads, channels, dropout = 0.0, pos_enc = True):
        super().__init__()
        self.pos_enc = pos_enc

        self.mha = MultiHeadAttention(heads, channels, dropout)

        #VERIFY
        self.conv_S = nn.Sequential( 
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.conv_Y = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.block_Z = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
        )


    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()


        if self.pos_enc:
            S = S + positional_encoding_2d(Sc, Sh, Sw, device=S.device)
            Y = Y + positional_encoding_2d(Yc, Yh, Yw, device=Y.device)

        V = self.conv_S(S).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        KQ = self.conv_Y(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)

        Z = self.mha(KQ, KQ, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        del KQ, V, Yb, Sc, Yh, Yw

        Z = self.block_Z(Z)

        Z =  Z * S
        del S

        return Z



class UNet_2D(nn.Module):
    
    
    def __init__(self, in_channels=5,fpa_block=False,sa=False,deep_supervision=False, mhca=False, respath=False):
        super(UNet_2D, self).__init__()

        self.fpa_block = fpa_block
        self.sa = sa
        self.deep_supervision = deep_supervision
        self.mhca = mhca
        self.respath = respath

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # define extra modules here
        self.mhsa = MHSABlock(8 , 512)
        
        self.fpa = FPABlock(256,256)

        self.mhcal4 = MHCABlock(1,256)
        self.mhcal3 = MHCABlock(1,128)
        self.mhcal2 = MHCABlock(1,64)

        self.respath1 = ResPath(32,32,4)
        self.respath2 = ResPath(64,64,3)
        self.respath3 = ResPath(128,128,2)
        self.respath4 = ResPath(256,256,1)

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

        
        # enable multi head self attention module
        if self.sa:
            x = self.pool(x)
            x = self.doubleconv5(x)
            x = self.mhsa(x)
            x = self.up1(x)
        
        else:
            x = self.pool(x)
            x = self.doubleconv5(x)
            x = self.up1(x)
        
        #### DECODER ####
        
        #BLOCK 1

        if self.respath:
            skip_4 = self.respath4(skip_4)
        
        if self.mhca:
            skip_4 = self.mhcal4(x,skip_4)
        
        x = torch.cat((x, skip_4), dim=1) # skip layer
        x = self.doubleconv6(x)

        if self.training and self.deep_supervision:
            x1 = self.output1(x)
        
        #BLOCK 2

        if self.respath:
            skip_3 = self.respath3(skip_3)

        if self.mhca:
            skip_3 = self.mhcal3(x,skip_3)

        x = self.up2(x)
        x = torch.cat((x, skip_3), dim=1) # skip layer
        x = self.doubleconv7(x)

        if self.training and self.deep_supervision:
            x2 = self.output2(x)


        #BLOCK 3

        if self.respath:
            skip_2 = self.respath2(skip_2)

        if self.mhca:
            skip_2 = self.mhcal2(x,skip_2)
        
        
        x = self.up3(x)
        x = torch.cat((x, skip_2), dim=1) # skip layer
        x = self.doubleconv8(x)

        if self.training and self.deep_supervision:
            x3 = self.output3(x)

        
        
        #BLOCK 4

        if self.respath:
            skip_1 = self.respath1(skip_1)

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