import torch.nn as nn
import torch
from thop import profile
from model.DilateFormer import *
from fvcore.nn import FlopCountAnalysis

def weights_init_normal(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

class DenseBlock(nn.Module):
    def __init__(self, in_channel, k, num_module=4):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_module):
            layer.append(self.conv_block(
                k * i + in_channel, k))
        self.net = nn.Sequential( * layer)
    def conv_block(self, input_channels, k):
        return nn.Sequential(
            nn.BatchNorm2d(input_channels), nn.LeakyReLU(),
            nn.Conv2d(input_channels, k, kernel_size=3, padding=1))  
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim = 1)
        return X

class Conv_path(nn.Module):
    def __init__(self, in_channel=32, k=8):
        super(Conv_path, self).__init__()     
        self.Dense = DenseBlock(in_channel=in_channel, k=k)
        self.final_conv = nn.Conv2d(4*k+ in_channel, 32, 1)
    def forward(self, x):
        x1 = self.Dense(x)     
        x2 = self.final_conv(x1)
        return x2

class Fuse_MSDA(nn.Module):
    def __init__(self, H, W, Ph, Pw, in_chans, embed_dim, hidden_dim,
                 depths, num_heads, kernel_size, dilation, shallow_dim=32):
        super(Fuse_MSDA,self).__init__()
        #########   Trans Path   ########## 
        self.Trans_path =  Dilateformer_MSDA(
                            H=H, W=W, Ph=Ph, Pw=Pw, in_chans=in_chans, 
                            embed_dim=embed_dim, hidden_dim=hidden_dim,
                            depths=depths, num_heads=num_heads, kernel_size=kernel_size, dilation=dilation)
        #########   Conv Path   ########## 
        self.Conv_path = Conv_path()
        self.fuse = nn.Sequential(nn.Conv2d(shallow_dim*2, shallow_dim, kernel_size=1))
    
    def forward(self, x):
        x1 = self.Trans_path (x)
        x_trans = x1 + x
        x2 = self.Conv_path(x)
        x_conv = x2 + x
        x3 = torch.cat((x_trans,x_conv), dim=1)      
        x4 = self.fuse(x3)
        return x4

class Fuse_MHSA(nn.Module):
    def __init__(self, H, W, Ph, Pw, in_chans, embed_dim, hidden_dim,
                 depths, num_heads, shallow_dim=32):
        super(Fuse_MHSA,self).__init__()
        #########   Trans Path   ########## 
        self.Trans_path =  Dilateformer_MHSA(
                            H=H, W=W, Ph=Ph, Pw=Pw, in_chans=in_chans, 
                            embed_dim=embed_dim, hidden_dim=hidden_dim,
                            depths=depths, num_heads=num_heads)
        #########   Conv Path   ########## 
        self.Conv_path = Conv_path()
        self.fuse = nn.Sequential(nn.Conv2d(shallow_dim*2, shallow_dim, kernel_size=1))
    
    def forward(self, x):
        x1 = self.Trans_path (x)
        x_trans = x1 + x
        x2 = self.Conv_path(x)
        x_conv = x2 + x   
        x3 = torch.cat((x_trans,x_conv), dim=1)      
        x4 = self.fuse(x3)
        return x4

class Generator(nn.Module):
    def __init__(self, in_channel=1, shallow_dim=32, num_layers=6, H=20, W=1024, Ph=2, Pw=4, embed_dim=96, hidden_dim=16,
                 depths=[1, 1, 1, 1, 1, 1], num_heads=[3, 3, 3, 3, 3, 3], dilate_attention=[True, True, True, False, False, False],
                 kernel_size=3, dilation=[1, 2, 3]):
        super(Generator,self).__init__()

        ######### Initial Feature layer   ##########
        self.shallow_feature = nn.Sequential(nn.Conv2d(in_channel, shallow_dim, 3, 1, 1),
                                        nn.BatchNorm2d(shallow_dim), nn.LeakyReLU())     #  padding = (kernel_size - 1) // 2       
        #########  Deep Feature layer   ##########
        self.stages = nn.ModuleList()
        for i_layer in range(num_layers):
            if dilate_attention[i_layer]:
                stage = Fuse_MSDA(H=H, 
                                  W=W, 
                                  Ph=Ph, 
                                  Pw=Pw, 
                                  in_chans=shallow_dim, 
                                  embed_dim=embed_dim, 
                                  hidden_dim=hidden_dim,
                                  depths=[depths[i_layer]], 
                                  num_heads=[num_heads[i_layer]], 
                                  kernel_size=kernel_size, 
                                  dilation=dilation
                                  )
            else:
                stage = Fuse_MHSA(H=H, 
                                  W=W, 
                                  Ph=Ph, 
                                  Pw=Pw, 
                                  in_chans=shallow_dim, 
                                  embed_dim=embed_dim, 
                                  hidden_dim=hidden_dim,
                                  depths=[depths[i_layer]], 
                                  num_heads=[num_heads[i_layer]], 
                                  )
            self.stages.append(stage)

        #########   out Layer   ########## 
        self.out_layer = nn.Sequential(nn.BatchNorm2d(shallow_dim), nn.LeakyReLU(),nn.Conv2d(shallow_dim, in_channel, 3, 1, 1))
            
    def forward(self, x):
      
        x1 = self.shallow_feature(x)   # torch.Size([32, 32, 20, 1024])
        for stage in self.stages:
            x1 = stage(x1)
        out = self.out_layer(x1)

        mask = torch.zeros_like(x)
        mask[x == 0] = 1
        output = torch.mul(mask, out) + x           
        return output


if __name__ == "__main__":
    
    X = torch.rand(size=(16, 1, 20, 1024))
    net = Generator()
    out = net(X)
    print(out.shape)


