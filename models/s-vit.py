import math
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
from collections import OrderedDict

sys.path.append('unet/unet_transformer')
from unet import TransUnet

v = TransUnet(in_channels=5,img_dim=256,vit_blocks=1,vit_dim_linear_mhsa_block=512,classes=1)



img = torch.randn(4, 5, 256, 256)





if torch.cuda.is_available():
    net = v.cuda()
    CT = img.cuda()

    
#img = L_init(img)

print(img.shape)

pred = net(CT)
#
print(pred.shape)