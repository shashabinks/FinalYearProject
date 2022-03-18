import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import timm


class Attention(nn.Module):
    def __init__(self,dim,n_heads=12,qkv_bias=False,attn_p=0.,proj_p = 0.):
        super(Attention,self).__init__()
        self.dim = dim                            #dimension of each patch of image
        self.n_heads = n_heads                    #number of heads in MHA
        self.head_dim = dim//n_heads              #the lenght of q,k,v for each patch
        
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim,dim*3,bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_p)
        
    def forward(self,x):
        n_samples,n_tokens,dim = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim)
        qkv = qkv.permute(2,0,3,1,4)
        
        q,k,v = qkv[0],qkv[1],qkv[2]
        
        k_t = k.transpose(-2,-1)
        
        dp = (q@k_t)*self.scale
        
        attn = dp.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn@v
        weighted_avg = weighted_avg.transpose(1,2)
        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        
        return x
        


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,p=0.):
        super(MLP,self).__init__()
        self.fc1 = nn.Linear(in_features,hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features,out_features)
        self.drop = nn.Dropout(p)
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x 



class Block(nn.Module):
    def __init__(self,dim,n_heads,mlp_ratio=4,qkv_bias=True,p=0.,attn_p=0.):
        super(Block,self).__init__()
        self.norm1 = nn.LayerNorm(dim,eps=1e-6)
        self.attn = Attention(dim=dim,n_heads = n_heads,qkv_bias = qkv_bias,attn_p = attn_p,proj_p = p)
        self.norm2 = nn.LayerNorm(dim,eps=1e-6)
        hidden_features = int(dim*mlp_ratio) #3072
        self.mlp = MLP(in_features = dim,hidden_features = hidden_features,out_features=dim)
    
    def forward(self,x):
        
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x


def conv_trans_block(in_channels,out_channels):
    conv_trans_block = nn.Sequential(
    nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1),
    nn.ReLU(inplace=True),
    nn.BatchNorm2d(out_channels)
    )
    return conv_trans_block





class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    

class ResNetV2(nn.Module):

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(5, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]




class Embeddings(nn.Module):

    def __init__(self,embed_dim = 768,n_patches=196,p=0.,in_channels=3):
        super(Embeddings, self).__init__()

        # this part is based on ResNet, maybe use a different encoder?
        self.hybrid_model = ResNetV2(block_units=(3, 4, 9), width_factor=1)

        in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=embed_dim,
                                       kernel_size=1,
                                       stride=1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.dropout = nn.Dropout(p=p)


    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features



class transUnet(nn.Module):
    def __init__(self,img_size = 256,patch_size=16,in_channels = 5,n_classes=1,embed_dim = 768,depth=12,n_heads=12,mlp_ratio=4.,qkv_bias=True,p=0.,attn_p=0.):
        super(transUnet,self).__init__()
        
        self.embed_dim = embed_dim
        self.img_size = img_size
        
        self.batch_norm = nn.BatchNorm2d(5)
        self.conv_2d  = nn.Conv2d(5,3,(1,1))

        self.L_init = nn.Sequential(self.batch_norm,self.conv_2d)

        

        
        self.n_patches = (img_size//patch_size) ** 2

        # encoder part and splitting into patches
        self.embeddings = Embeddings(embed_dim,self.n_patches,p)

        
        # vit
        self.blocks = nn.ModuleList([Block(dim = embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias = qkv_bias, p=p, attn_p=attn_p,) for _ in range(depth)])

        
        #bottleneck layer
        self.deconv1 = conv_trans_block(embed_dim,512)
        
        self.deconv2_1 = conv_trans_block(1024,256)
        self.deconv2_2 = conv_trans_block(256,256)
        
        self.deconv3_1 = conv_trans_block(512,128)
        self.deconv3_2 = conv_trans_block(128,128)
        
        
        self.deconv4_1 = conv_trans_block(192,64)
        self.deconv4_2 = conv_trans_block(64,64)
        
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        self.prefinal_1 = conv_trans_block(64,32)
        self.prefinal_2 = conv_trans_block(32,32)
        
        self.out = nn.Conv2d(32,1,kernel_size=1)
        
                        
        
    def forward(self,x):
        
        n_samples = x.shape[0]
        

        #x = self.L_init(x)

        x = self.embeddings(x)
        projections = x[0]
        features = x[1]

        for block in self.blocks:
            projections = block(projections)
        
        projections = projections.transpose(1,2)
        projections = projections.reshape(n_samples,self.embed_dim,int(self.img_size/16),int(self.img_size/16))
        
        x1 = projections
        
        x1 = self.deconv1(x1)     #(n,512,224/16,224/16)

        x1 = self.upsample(x1)    #(n,512,224/8,224/8)
        x1 = self.deconv2_1(torch.cat([x1,features[0]],1))
        x1 = self.deconv2_2(x1)
        
        
        x1 = self.upsample(x1)    #(n,256,224/4,224/4)

        x1 = self.deconv3_1(torch.cat([x1,features[1]],1))
        x1 = self.deconv3_2(x1)
        
        x1 = self.upsample(x1)    #(n,128,224/2,224/2)
        
        x1 = self.deconv4_1(torch.cat([x1,features[2]],1))
        x1 = self.deconv4_2(x1)
        
        x1 = self.upsample(x1)   #(n,64,224,224)
        x1 = self.prefinal_1(x1)
        x1 = self.prefinal_2(x1)
        
        x1 = self.out(x1)
        #x1 = torch.sigmoid(x1)
        return x1


if __name__ == "__main__":
    batch_size = 4
    num_classes = 5
    initial_kernels = 32

    net = transUnet()
    
    # torch.save(net.state_dict(), 'model.pth')
    CT = torch.randn(batch_size, 5, 256, 256)    # Batchsize, modal, hight,

    print("Input:", CT.shape)
    if torch.cuda.is_available():
        net = net.cuda()
        CT = CT.cuda()

    segmentation_prediction = net(CT)
    print("Output:",segmentation_prediction.shape)