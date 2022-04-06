import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math
from torchsummary import summary

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

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
        nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
        
        nn.Conv2d(32, n_classes, kernel_size=1)
    )


def output2(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
        nn.Conv2d(32, n_classes, kernel_size=1)
    )


def output3(n_classes):
    return nn.Sequential(
        nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
        nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
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

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))# self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out =self.shared_MLP(self.max_pool(x))# self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
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

class MHCABlock(nn.Module):
    def __init__(self, heads, channels, dropout = 0.0, pos_enc = True,last_layer=False):
        super().__init__()
        self.pos_enc = pos_enc

        self.mha = MultiHeadAttention(heads, channels, dropout)
        self.last_layer = last_layer

        #VERIFY
        self.conv_S = nn.Sequential( 
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.conv_Y = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.conv_Y1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False), 
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.block_Z = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
            nn.ConvTranspose2d(channels, channels, kernel_size=2, stride=2),
        )

        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, Y, S):
        Sb, Sc, Sh, Sw = S.size()
        Yb, Yc, Yh, Yw = Y.size()

        

        if self.pos_enc:
            S = S + positional_encoding_2d(Sc, Sh, Sw, device=S.device)
            Y = Y + positional_encoding_2d(Yc, Yh, Yw, device=Y.device)

        V = self.conv_S(S).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)

        if self.last_layer:
            
            KQ = self.conv_Y1(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)
        else:
            KQ = self.conv_Y(Y).reshape(Yb, Sc, Yh*Yw).permute(0, 2, 1)

        Z = self.mha(KQ, KQ, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        del KQ, V, Yb, Sc, Yh, Yw

        Z = self.block_Z(Z)

        Z =  Z * S

        return Z

    
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(F_l, F_l, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_l),
            nn.ReLU(inplace=True)
        )

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g = self.up(g)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out



class AA_UNet(nn.Module):
    """Shallow Unet with ResNet18 or ResNet34 encoder.
    """

    def __init__(self, *, pretrained=False, out_channels=1,freeze=False, fpa_block=False, respaths=False,deep_supervision=False,mhca=False):
        super().__init__()
        self.encoder = torchvision.models.resnet34(pretrained=pretrained)

        self.mhca = mhca
        self.fpa_block = fpa_block
        self.deep_supervision = deep_supervision
        self.respath = respaths
        

        # define extra modules

        self.respath1 = CBAM(64)
        self.respath2 = CBAM(64)
        self.respath3 = CBAM(128)
        self.respath4 = CBAM(256)

        self.fpa = FPABlock(512,512)
    
        self.encoder_layers = list(self.encoder.children())
        self.encoder.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = self.encoder.conv1

        self.block1 = nn.Sequential(*self.encoder_layers[1:3])
        self.block2 = nn.Sequential(*self.encoder_layers[3:5])
        self.block3 = self.encoder_layers[5]
        self.block4 = self.encoder_layers[6]
        self.block5 = self.encoder_layers[7]

        self.mhca1 = MHCABlock(1,256)
        self.mhca2 = MHCABlock(1,128)
        self.mhca3 = MHCABlock(1,64)
        self.mhca4 = MHCABlock(1,64,last_layer=True)

    
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

        if self.respath:
            block1 = self.respath1(block1)
            block2 = self.respath2(block2)
            block3 = self.respath3(block3)
            block4 = self.respath4(block4)


        # bottleneck
        block5 = self.block5(block4)

        if self.fpa_block:
            x = self.fpa(block5)

        if self.mhca:
            block4 = self.mhca1(x,block4)

    
        x = self.up_conv6(block5)

        # level 4
        
        x = torch.cat([x, block4], dim=1) 
        x = self.conv6(x)

    
        if self.training and self.deep_supervision:
            x1 = self.output1(x)
            

        if self.mhca:
            block3 = self.mhca2(x,block3)

        x = self.up_conv7(x)

        # level 3 
        
        x = torch.cat([x, block3], dim=1)
        x = self.conv7(x)

        if self.training and self.deep_supervision:
            x2 = self.output2(x)

        if self.mhca:
            block2 = self.mhca3(x,block2)

        x = self.up_conv8(x)

        # level 2

        x = torch.cat([x, block2], dim=1)
        x = self.conv8(x)

        if self.training and self.deep_supervision:
            x3 = self.output3(x)

        if self.mhca:
            block1 = self.mhca4(x,block1)


        x = self.up_conv9(x)

        # level 1
    
        x = torch.cat([x, block1], dim=1)
        x = self.conv9(x)

        if self.training and self.deep_supervision:
            x4 = self.output4(x)

        x = self.up_conv10(x)

        # level output
        x = self.conv10(x)

        if self.training and self.deep_supervision:
            
            return [x, x4, x3, x2, x1]

        return x

if __name__ == "__main__":
    batch_size = 8
    num_classes = 5
    initial_kernels = 32

    net = AA_UNet(fpa_block=True,respaths=True,deep_supervision=False,mhca=True)
    
    
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)

    print(segmentation_prediction.shape)
    



    
    