import nibabel as nib
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import math


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
          
        for m in self.children():
          init_weights(m, init_type='kaiming')
    def forward(self, x):
        x = self.conv(x)
        return x

# o = (i -1)*s - 2*p + k + output_padding
class transposed_conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(transposed_conv2d, self).__init__()

        self.conv = nn.Sequential(
          nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2),
          nn.BatchNorm2d(out_ch)
        )

    def forward(self,x):
      x = self.conv(x)
      return x

class MultiResBlock(nn.Module):
    def __init__(self,in_ch,U,alpha=1):
      super(MultiResBlock,self).__init__()

      W = alpha*U 
     
      filters = [int(W*0.125) + int(W*0.375) +int(W*0.5), int(W*0.125),int(W*0.375),int(W*0.5)]
      #filters = [W,int(W*0.25),int(W*0.25),int(W*0.5)]
      self.shortcut = conv2d_block(in_ch,filters[0],1,activation=False)
      self.conv3 = conv2d_block(in_ch,filters[1],3)
      self.conv5 = conv2d_block(filters[1],filters[2],3)
      self.conv7 = conv2d_block(filters[2],filters[3],3)
      self.B1 = nn.BatchNorm2d(filters[0])
      self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(filters[0])
        )
    
    def forward(self,x):
      shortcut = self.shortcut(x)
      conv3 = self.conv3(x)
      conv5 = self.conv5(conv3)
      conv7 = self.conv7(conv5)
      out = torch.cat([conv3,conv5,conv7],dim=1)
      out = self.B1(out)
      out = torch.add(shortcut,out)
      out = self.final(out)
      return out

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

def init_weights(net, init_type='kaiming'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class MultiResNet(nn.Module):
   
    def __init__(self, n_classes=1, in_channel=4):
       super(MultiResNet, self).__init__()
       
       f=32
       filters = [f, f*2, f*4, f*8, f*16]
       

       self.mresblock1 = MultiResBlock(in_channel,filters[0])
       self.maxpool1 = nn.MaxPool2d(kernel_size=2)
       self.respath1 = ResPath(filters[0],filters[0],4)

       self.mresblock2 = MultiResBlock(filters[0],filters[1])
       self.maxpool2 = nn.MaxPool2d(kernel_size=2)
       self.respath2 = ResPath(filters[1],filters[1],3)

       self.mresblock3 = MultiResBlock(filters[1],filters[2])
       self.maxpool3 = nn.MaxPool2d(kernel_size=2)
       self.respath3 = ResPath(filters[2],filters[2],2)

       self.mresblock4 = MultiResBlock(filters[2],filters[3])
       self.maxpool4 = nn.MaxPool2d(kernel_size=2)
       self.respath4 = ResPath(filters[3],filters[3],1)

       self.mresblock5 = MultiResBlock(filters[3],filters[4])
       self.mhsa = MHSABlock(1,filters[4])

       self.up4 = transposed_conv2d(filters[4],filters[3],2)
       self.mresblock6 = MultiResBlock(filters[4],filters[3])

       self.up3 = transposed_conv2d(filters[3],filters[2],2)
       self.mresblock7 = MultiResBlock(filters[3],filters[2])

       self.up2 = transposed_conv2d(filters[2],filters[1],2)
       self.mresblock8 = MultiResBlock(filters[2],filters[1])

       self.up1 = transposed_conv2d(filters[1],filters[0],2)
       self.mresblock9 = MultiResBlock(filters[1],filters[0])

       self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1,bias=True)

    def forward(self,x):
       conv1 = self.mresblock1(x)
       maxpool1 = self.maxpool1(conv1)
       conv1 = self.respath1(conv1)

       conv2 = self.mresblock2(maxpool1)
       maxpool2 = self.maxpool2(conv2)
       conv2 = self.respath2(conv2)

       conv3 = self.mresblock3(maxpool2)
       maxpool3 = self.maxpool3(conv3)
       conv3 = self.respath3(conv3)

       conv4 = self.mresblock4(maxpool3)
       maxpool4 = self.maxpool4(conv4)
       conv4 = self.respath4(conv4)

       center = self.mresblock5(maxpool4) # bottleneck, maybe add a attention module here?

       up4 = torch.cat((self.up4(center),conv4),dim=1)
       up4 = self.mresblock6(up4)

       up3 = torch.cat((self.up3(up4),conv3),dim=1)
       up3 = self.mresblock7(up3)

       up2 = torch.cat((self.up2(up3),conv2),dim=1)
       up2 = self.mresblock8(up2)

       up1 = torch.cat((self.up1(up2),conv1),dim=1)
       up1 = self.mresblock9(up1)

       final = self.final(up1)

       return final

if __name__ == "__main__":
    batch_size = 4
    num_classes = 1  # one hot
    initial_kernels = 32
    
    
    
    net = MultiResNet()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 4, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()
        torch.cuda.empty_cache()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)