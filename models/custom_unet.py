
# A Dense Multi-Modal U-Net
import numpy as np
import torch
import torch.nn as nn
import math

class MultiHeadDense(nn.Module):
    def __init__(self, d, bias=False):
        super(MultiHeadDense, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(d, d))
        if bias:
            raise NotImplementedError()
            self.bias = Parameter(torch.Tensor(d, d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # x:[b, h*w, d]
        b, wh, d = x.size()
        x = torch.bmm(x, self.weight.repeat(b, 1, 1))
        # x = F.linear(x, self.weight, self.bias)

        del b

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()

    def positional_encoding_2d(self, d_model, height, width):
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
        pe = torch.zeros(d_model, height, width)
        try:
            pe = pe.to(torch.device("cuda:0"))
        except RuntimeError:
            pass
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        return pe

    def forward(self, x):
        raise NotImplementedError()


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels / 2))
        self.channels = channels
        inv_freq = 1. / (10000
                         **(torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x,
                             device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y,
                             device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()),
                          dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2),
                          device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)        
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)


class MultiHeadSelfAttention(MultiHeadAttention):
    def __init__(self, channel):
        super(MultiHeadSelfAttention, self).__init__()
        self.query = MultiHeadDense(channel, bias=False)
        self.key = MultiHeadDense(channel, bias=False)
        self.value = MultiHeadDense(channel, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.pe = PositionalEncodingPermute2D(channel)

    def forward(self, x):
        b, c, h, w = x.size()
        # pe = self.positional_encoding_2d(c, h, w)
        pe = self.pe(x)
        x = x + pe
        x = x.reshape(b, c, h * w).permute(0, 2, 1)  #[b, h*w, d]
        Q = self.query(x)
        K = self.key(x)
        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(c))  #[b, h*w, h*w]

        del Q , K

        V = self.value(x)
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(b, c, h, w)

        del A,V

        return x


class MultiHeadCrossAttention(MultiHeadAttention):
    def __init__(self, channelY, channelS):
        super(MultiHeadCrossAttention, self).__init__()
        self.Sconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.Yconv = nn.Sequential(
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.query = MultiHeadDense(channelS, bias=False)
        self.key = MultiHeadDense(channelS, bias=False)
        self.value = MultiHeadDense(channelS, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(channelS, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.Yconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(channelY, channelY, kernel_size=3, padding=1),
            nn.Conv2d(channelY, channelS, kernel_size=1),
            nn.BatchNorm2d(channelS), nn.ReLU(inplace=True))
        self.softmax = nn.Softmax(dim=1)
        self.Spe = PositionalEncodingPermute2D(channelS)
        self.Ype = PositionalEncodingPermute2D(channelY)

    def forward(self, Y, S):
        _, Sc, _, _ = S.size()
        Yb, _, Yh, Yw = Y.size()
        # Spe = self.positional_encoding_2d(Sc, Sh, Sw)
        Spe = self.Spe(S)
        S = S + Spe
        S1 = self.Sconv(S).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        V = self.value(S1)
        # Ype = self.positional_encoding_2d(Yc, Yh, Yw)

        del S1

        Ype = self.Ype(Y)
        Y = Y + Ype
        Y1 = self.Yconv(Y).reshape(Yb, Sc, Yh * Yw).permute(0, 2, 1)
        Y2 = self.Yconv2(Y)

        del Y

        Q = self.query(Y1)
        K = self.key(Y1)

        del Y1

        A = self.softmax(torch.bmm(Q, K.permute(0, 2, 1)) / math.sqrt(Sc))
        x = torch.bmm(A, V).permute(0, 2, 1).reshape(Yb, Sc, Yh, Yw)

        del Q, K, A, V , Yb, Sc, Yh, Yw

        Z = self.conv(x)

        del x

        Z = Z * S

        del S

        Z = torch.cat([Z, Y2], dim=1)

        del Y2

        return Z

def croppCenter(tensorToCrop,finalShape):
    org_shape = tensorToCrop.shape

    diff = np.zeros(2)
    diff[0] = org_shape[2] - finalShape[2]
    diff[1] = org_shape[3] - finalShape[3]

    croppBorders = np.zeros(2,dtype=int)
    croppBorders[0] = int(diff[0]/2)
    croppBorders[1] = int(diff[1]/2)

    return tensorToCrop[:, :,
                        croppBorders[0]:croppBorders[0] + finalShape[2],
                        croppBorders[1]:croppBorders[1] + finalShape[3]]

class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
            super(ConvBlock2d, self).__init__()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
                
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ConvTrans2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvTrans2d, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            
        )

    def forward(self, x):
        x = self.conv1(x)
        return x 

class UpBlock2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock2d, self).__init__()
        self.up_conv = ConvTrans2d(in_ch, out_ch)
        self.conv = ConvBlock2d(2 * out_ch, out_ch)

    def forward(self, x, down_features):
        x = self.up_conv(x)
        x = torch.cat([x, down_features], dim=1)
        x = self.conv(x)
        return x


class TDMM_Unet_4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_of_features = 32):
        super(TDMM_Unet_4,self).__init__()

        self.in_dim = in_channels
        self.out_dim = num_of_features
        self.final_out_dim = out_channels


        # ~~~ ENCODING PATHS ~~~ #

        # 4 image modalities, therefore 5 separate paths, all the same structure

        # Encoder ~ M1 ~ CBF
        self.down_1_0 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_0 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_0 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_4_0 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_0 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M2 ~ CBV
        self.down_1_1 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_1 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_1 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_1 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M3 ~ Tmax
        self.down_1_2 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_2 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_2 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_2 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder ~ M4 ~ MTT
        self.down_1_3 = ConvBlock2d(self.in_dim, self.out_dim)
        self.pool_1_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_2_3 = ConvBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.pool_2_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_3_3 = ConvBlock2d(self.out_dim * 12, self.out_dim * 4)
        self.pool_3_3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down_4_3 = ConvBlock2d(self.out_dim * 28, self.out_dim * 8)
        self.pool_4_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        

        # ~~~ BRIDGE ~~~ #
        self.bridge = ConvBlock2d(self.out_dim * 60, self.out_dim * 16 )
        self.mhsa = MultiHeadSelfAttention(self.out_dim * 16)

        # ~~~ DECODER PATH ~~~ #

        

        self.mhca1 = MultiHeadCrossAttention(self.out_dim * 16,self.out_dim * 8)
        self.conv1 = nn.Conv2d(self.out_dim * 16, self.out_dim * 8, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(self.out_dim * 8)

        self.mhca2 = MultiHeadCrossAttention(self.out_dim * 8,self.out_dim * 4)
        self.conv2 = nn.Conv2d(self.out_dim * 8, self.out_dim * 4, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(self.out_dim * 4)

        self.mhca3 = MultiHeadCrossAttention(self.out_dim * 4,self.out_dim * 2)
        self.conv3 = nn.Conv2d(self.out_dim * 4, self.out_dim * 2, 3, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(self.out_dim * 2)

        self.mhca4 = MultiHeadCrossAttention(self.out_dim * 2 ,self.out_dim * 1)
        self.conv4 = nn.Conv2d(self.out_dim * 2, self.out_dim * 1, 3, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(self.out_dim * 1)

        self.upLayer1 = UpBlock2d(self.out_dim * 16, self.out_dim * 8)
        self.upLayer2 = UpBlock2d(self.out_dim * 8, self.out_dim * 4)
        self.upLayer3 = UpBlock2d(self.out_dim * 4, self.out_dim * 2)
        self.upLayer4 = UpBlock2d(self.out_dim * 2, self.out_dim * 1)


        # ~~~ OUTPUT ~~~ #
        self.out = nn.Conv2d(self.out_dim, self.final_out_dim, kernel_size=3, stride=1, padding=1)
    

    def forward(self, input):

        # define all 5 image modalities
        # each image is defined as: batch_sze x modal x height x width
        m1 = input[:,0:1,:,:]
        m2 = input[:,1:2,:,:]
        m3 = input[:,2:3,:,:]
        m4 = input[:,3:4,:,:]
        


        # ~~~ ENCODING ~~~ #
        # ~~~ L1 ~~~ #
        down_1_0 = self.down_1_0(m1)  # Batch Size * outdim * volume_size * height * width
        down_1_1 = self.down_1_1(m2)
        down_1_2 = self.down_1_2(m3)
        down_1_3 = self.down_1_3(m4)
        


        # ~~~ L2 ~~~ #
        
        # prepare the input
        input_2nd_0 = torch.cat((self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3)), dim=1)

        input_2nd_1 = torch.cat((self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0)), dim=1)

        input_2nd_2 = torch.cat((self.pool_1_2(down_1_2),
                                 self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1)), dim=1)

        input_2nd_3 = torch.cat((self.pool_1_3(down_1_3),
                                 self.pool_1_0(down_1_0),
                                 self.pool_1_1(down_1_1),
                                 self.pool_1_2(down_1_2)), dim=1)
        
        

        #print(input_2nd_0.shape)
        # do the convolution
        down_2_0 = self.down_2_0(input_2nd_0)
        down_2_1 = self.down_2_1(input_2nd_1)
        down_2_2 = self.down_2_2(input_2nd_2)
        down_2_3 = self.down_2_3(input_2nd_3)
        

        
        # ~~~ L3 ~~~ #
        down_2_0m = self.pool_2_0(down_2_0)
        down_2_1m = self.pool_2_0(down_2_1)
        down_2_2m = self.pool_2_0(down_2_2)
        down_2_3m = self.pool_2_0(down_2_3)
        

        input_3rd_0 = torch.cat((down_2_0m, down_2_1m, down_2_2m, down_2_3m), dim=1)
        input_3rd_0 = torch.cat((input_3rd_0, croppCenter(input_2nd_0, input_3rd_0.shape)), dim=1)

        input_3rd_1 = torch.cat((down_2_1m, down_2_2m, down_2_3m,down_2_0m), dim=1)
        input_3rd_1 = torch.cat((input_3rd_1, croppCenter(input_2nd_1, input_3rd_1.shape)), dim=1)

        input_3rd_2 = torch.cat((down_2_2m, down_2_3m,down_2_0m, down_2_1m), dim=1)
        input_3rd_2 = torch.cat((input_3rd_2, croppCenter(input_2nd_2, input_3rd_2.shape)), dim=1)

        input_3rd_3 = torch.cat((down_2_3m,down_2_0m, down_2_1m, down_2_2m), dim=1)
        input_3rd_3 = torch.cat((input_3rd_3, croppCenter(input_2nd_3, input_3rd_3.shape)), dim=1)

        

        down_3_0 = self.down_3_0(input_3rd_0)
        down_3_1 = self.down_3_1(input_3rd_1)
        down_3_2 = self.down_3_2(input_3rd_2)
        down_3_3 = self.down_3_3(input_3rd_3)
        

        # ~~~ L4 ~~~ #
        down_3_0m = self.pool_2_0(down_3_0)
        down_3_1m = self.pool_2_0(down_3_1)
        down_3_2m = self.pool_2_0(down_3_2)
        down_3_3m = self.pool_2_0(down_3_3)
        

        input_4th_0 = torch.cat((down_3_0m, down_3_1m, down_3_2m, down_3_3m), dim=1)
        input_4th_0 = torch.cat((input_4th_0, croppCenter(input_3rd_0, input_4th_0.shape)), dim=1)

        input_4th_1 = torch.cat((down_3_1m, down_3_2m, down_3_3m, down_3_0m), dim=1)
        input_4th_1 = torch.cat((input_4th_1, croppCenter(input_3rd_1, input_4th_1.shape)), dim=1)

        input_4th_2 = torch.cat((down_3_2m, down_3_3m, down_3_0m, down_3_1m), dim=1)
        input_4th_2 = torch.cat((input_4th_2, croppCenter(input_3rd_2, input_4th_2.shape)), dim=1)

        input_4th_3 = torch.cat((down_3_3m, down_3_0m, down_3_1m, down_3_2m), dim=1)
        input_4th_3 = torch.cat((input_4th_3, croppCenter(input_3rd_3, input_4th_3.shape)), dim=1)

        

        down_4_0 = self.down_4_0(input_4th_0)
        down_4_1 = self.down_4_1(input_4th_1)
        down_4_2 = self.down_4_2(input_4th_2)
        down_4_3 = self.down_4_3(input_4th_3)
        

        
        # ~~~ BRIDGE ~~~ #
        down_4_0m = self.pool_4_0(down_4_0)
        down_4_1m = self.pool_4_0(down_4_1)
        down_4_2m = self.pool_4_0(down_4_2)
        down_4_3m = self.pool_4_0(down_4_3)
        

        inputBridge = torch.cat((down_4_0m, down_4_1m, down_4_2m, down_4_3m), dim=1)
        inputBridge = torch.cat((inputBridge, croppCenter(input_4th_0, inputBridge.shape)), dim=1)

        bridge = self.bridge(inputBridge)

        # after rebuild, apply mhsa module
        y = self.mhsa(bridge)

        # ~~~~~~ Decoding path ~~~~~~~  #
        skip_1 = (down_4_0 + down_4_1 + down_4_2 + down_4_3) / 4.0  # most bottom one
        skip_2 = (down_3_0 + down_3_1 + down_3_2 + down_3_3) / 4.0
        skip_3 = (down_2_0 + down_2_1 + down_2_2 + down_2_3) / 4.0
        skip_4 = (down_1_0 + down_1_1 + down_1_2 + down_1_3) / 4.0      # top one

        #"""
        x = self.mhca1(y,skip_1)
        x = self.conv1(x)
        x = self.norm1(x)

        x = self.mhca2(x,skip_2)
        x = self.conv2(x)
        x = self.norm2(x)

        x = self.mhca3(x,skip_3)
        x = self.conv3(x)
        x = self.norm3(x)

        #print("YO")
        
        """
        x = self.mhca4(x,skip_4)
        x = self.conv4(x)
        x = self.norm4(x)

        print("YO")
        """

        """
        x = self.upLayer1(y, skip_1)
        x = self.upLayer2(x, skip_2)
        x = self.upLayer3(x, skip_3)
        x = self.upLayer4(x, skip_4)
        """
        x = self.upLayer4(x, skip_4)
        
        return self.out(x)


if __name__ == "__main__":
    batch_size = 4
    num_classes = 1  # one hot
    initial_kernels = 32
    
    
    
    net = TDMM_Unet_4(1, num_classes)
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 4, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()
        torch.cuda.empty_cache()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)