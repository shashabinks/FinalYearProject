import torch.nn as nn
import torch
from einops import rearrange
from .transformer import ViT


# taken and adapted form torchvision repo
# https://github.com/pytorch/vision/blob/af97ec2f4c9daac091b9a87355c4f22d37488004/torchvision/models/resnet.py#L86

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                norm_layer(planes * self.expansion),
            )
        else:
            self.downsample = nn.Identity()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out

class SingleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(SingleConv, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            norm_layer(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    """
    Double convolution block that keeps that spatial sizes the same
    """

    def __init__(self, in_ch, out_ch, norm_layer=None):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(SingleConv(in_ch, out_ch, norm_layer),
                                  SingleConv(out_ch, out_ch, norm_layer))

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    """
    Doubles spatial size with bilinear upsampling
    Skip connections and double convs
    """

    def __init__(self, in_ch, out_ch):
        super(Up, self).__init__()
        mode = "bilinear"
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        """
        Args:
            x1: [b,c, h, w]
            x2: [b,c, 2*h,2*w]
        Returns: 2x upsampled double conv reselt
        """
        x = self.up(x1)
        if x2 is not None:
            x = torch.cat([x2, x], dim=1)
        return self.conv(x)

class TransUnet(nn.Module):
    def __init__(self, *, img_dim, in_channels, classes,
                 vit_blocks=12,
                 vit_heads=4,
                 vit_dim_linear_mhsa_block=1024,
                 ):
        """
        Args:
            img_dim: the img dimension
            in_channels: channels of the input
            classes: desired segmentation classes
            vit_blocks: MHSA blocks of ViT
            vit_heads: number of MHSA heads
            vit_dim_linear_mhsa_block: MHSA MLP dimension
        """
        super().__init__()
        self.inplanes = 128
        vit_channels = self.inplanes * 8

        # Not clear how they used resnet arch. since the first input after conv
        # must be 128 channels and half spat dims.
        in_conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                             bias=False)
        bn1 = nn.BatchNorm2d(self.inplanes)
        self.init_conv = nn.Sequential(in_conv1, bn1, nn.ReLU(inplace=True))
        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.conv1 = Bottleneck(self.inplanes, self.inplanes * 2, stride=2)
        self.conv2 = Bottleneck(self.inplanes * 2, self.inplanes * 4, stride=2)
        self.conv3 = Bottleneck(self.inplanes * 4, vit_channels, stride=2)

        self.img_dim_vit = img_dim // 16
        self.vit = ViT(img_dim=self.img_dim_vit,
                       in_channels=vit_channels,  # encoder channels
                       patch_dim=1,
                       dim=vit_channels,  # vit out channels for decoding
                       blocks=vit_blocks,
                       heads=vit_heads,
                       dim_linear_block=vit_dim_linear_mhsa_block,
                       classification=False)
        
        
        self.vit_conv = SingleConv(in_ch=vit_channels, out_ch=512)
        # add a aspp module here?

        self.dec1 = Up(1024, 256)
        self.dec2 = Up(512, 128)
        self.dec3 = Up(256, 64)
        self.dec4 = Up(64, 16)
        self.conv1x1 = nn.Conv2d(16, classes, kernel_size=1)

    def forward(self, x):
        # ResNet 50-like encoder
        x2 = self.init_conv(x)  # 128,64,64
        x4 = self.conv1(x2)  # 256,32,32
        x8 = self.conv2(x4)  # 512,16,16
        x16 = self.conv3(x8)  # 1024,8,8
        y = self.vit(x16)
        y = rearrange(y, 'b (x y) dim -> b dim x y ', x=self.img_dim_vit, y=self.img_dim_vit)
        y = self.vit_conv(y)
        y = self.dec1(y, x8)  # 256,16,16
        y = self.dec2(y, x4)
        y = self.dec3(y, x2)
        y = self.dec4(y)
        return self.conv1x1(y)


if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = TransUnet(in_channels=4,img_dim=256,vit_blocks=1,vit_dim_linear_mhsa_block=512,classes=1)
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 4, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)