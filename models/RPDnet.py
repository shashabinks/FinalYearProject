import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

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

def output4(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
        nn.Conv2d(32, n_classes, kernel_size=1)
    )

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

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size=2, stride=2
    )


class RPDNet(nn.Module):
    """Shallow Unet with ResNet18 or ResNet34 encoder.
    """

    def __init__(self, *, pretrained=False, out_channels=1,freeze=False, fpa_block=False, respaths=False, deep_supervision=False):
        super().__init__()
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.fpa_block = fpa_block
        self.deep_supervision = deep_supervision
        self.respath = respaths

        # define extra modules

        self.respath1 = ResPath(64,64,4)
        self.respath2 = ResPath(64,64,3)
        self.respath3 = ResPath(128,128,2)
        self.respath4 = ResPath(256,256,1)

        self.fpa = FPABlock(512,512)

    
        self.encoder_layers = list(self.encoder.children())
        self.encoder.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = self.encoder.conv1

        self.block1 = nn.Sequential(*self.encoder_layers[1:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

    
        # freeze weights of last block
        if freeze:
            for child in self.encoder_layers[7].children():
                for param in child.parameters():
                    param.requires_grad = False
                    

        self.up_conv6 = up_conv(512, 512)
        self.conv6 = double_conv(512 + 256, 256)

        self.up_conv7 = up_conv(256, 256)
        self.conv7 = double_conv(256 + 128, 128)

        self.up_conv8 = up_conv(128, 128)
        self.conv8 = double_conv(128 + 64, 64)

        self.up_conv9 = up_conv(64, 64)
        self.conv9 = double_conv(64 + 64, 32)

        # very final layer
        self.up_conv10 = up_conv(32, 32)
        self.conv10 = nn.Conv2d(32, out_channels, kernel_size=1)

        if not pretrained:
            self._weights_init()
        
        self.output1 = output1(1)
        self.output2 = output2(1)
        self.output3 = output3(1)
        self.output4 = output4(1)

    def _weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        # bottleneck
        block5 = self.block5(block4)

        if self.fpa_block:
            x = self.fpa(block5)
        
        x = self.up_conv6(block5)

        
        # level 4
        if self.respath:
            block4 = self.respath4(block4)


        x = torch.cat([x, block4], dim=1)
        x = self.conv6(x)

        if self.training and self.deep_supervision:
            x1 = self.output1(x)

        x = self.up_conv7(x)


        # level 3
        if self.respath:
            block3 = self.respath3(block3)

        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        if self.training and self.deep_supervision:
            x2 = self.output2(x)

        x = self.up_conv8(x)

        # level 2
        if self.respath:
            block2 = self.respath2(block2)

        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        if self.training and self.deep_supervision:
            x3 = self.output3(x)


        x = self.up_conv9(x)

        # level 1
        if self.respath:
            block1 = self.respath1(block1)

        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        if self.training and self.deep_supervision:
            x4 = self.output4(x)

        x = self.up_conv10(x)

        # level output
        x = self.conv10(x)

        if self.training and self.deep_supervision:
            return x, x4, x3, x2, x1

        return x

if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = RPDNet(fpa_block=True,respaths=True)
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size,5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)