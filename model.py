import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F

# each convolutional block in the encoding stage
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3) # kernel size here is 3, no padding is used
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


# the encoding stage, this is referred to as the contractive path
# each c-block is followed by a max pooling operation (between two block operations)

class Encoder(nn.Module):
    def __init__(self, chs=(3,64,128,256,512,1024)):        # we define the channel order, so we start from 3 channels and go down to 1024 channels (the very bottom level)
        super().__init__()                                  # this should be different for the lesion segmentation
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # we essentially loop through each channel and apply the feature in/outs
        self.pool = nn.MaxPool2d(2) # define a 2x2 pooling filter, can be 2x2x2 for the 3d case
    
    def forward(self,x):
        features = []
        for block in self.enc_blocks:       # for each block in the list of block operations
            x = block(x)                    # calculate the output given the input, x
            features.append(x)               # append the output features into the array (of each conv block), this is so that we can concatenate with the decoding path
            x = self.pool(x)
        return features

# the decoder stage is essentially the expansive path of the architecture
# we essentially do "up-convolution" or "transposed convolution" which halves the number of feature channels and concatenates with the corresponding cropped feature map
# from the contracting path (encoding stage)
# we also have to do cropping which is necessary due to the loss of border pixels in every convolution

class Decoder(nn.Module):
    def __init__(self, chs=(1024,512,256,128,64)):  # we go from a 1024 channel image back up to a 64 channel (eventually becomes 1 channel at the very end)
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])   # list of t-convolutions going upwards
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) # we do the same convolution operations as before
        # this contains the decoder blocks that perform two convs and relu operation
    
    def crop(self,enc_ftrs, x):
        _,_, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H,W])(enc_ftrs)
        return enc_ftrs
    
    def forward(self,x,encoder_features):
        for i in range(len(self.chs)-1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x) # crop the encoder features, preparing them for concatenation
            x = torch.cat([x,enc_ftrs], dim=1) # concatenate the feature from the encoder path with the output of the decoder block
            x = self.dec_blocks[i](x)          # x is the output, we input into the blocks for the decoder stage
        return x


# Now we just need to link all this up to form the overall u-net
class UNet(nn.Module):
    def __init__(self,enc_chs=(3,64,128,256,512,1024), dec_chs=(1024, 512, 256, 128, 64), num_class=1, retain_dim=False, out_sze=(572,572)):
        super().__init__()
        self.encoder = Encoder(enc_chs) # carry out the encoding
        self.decoder = Decoder(dec_chs) # carry out the decoding
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1) # the part at the very end, to produce the final segmented image
        self.retain_dim = retain_dim # boolean to decide whether we want to keep the size of the image as it was before
        self.out_size = out_sze # define the size of the output, this may be the same dimensionality as the input data if retaining
    
    def forward(self,x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out,self.out_size)     # make the output size the same as the input image size
        
        return out 


if __name__=="__main__":
    unet = UNet()
    x    = torch.randn(1, 3, 572, 572)
    print(unet(x).shape)