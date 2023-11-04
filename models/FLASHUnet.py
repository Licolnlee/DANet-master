from turtle import forward
import torch.nn as nn
import torch
from torch import Tensor, autograd, dropout
from torchvision.ops.deform_conv import DeformConv2d
# from modulated_deform_conv import *
# from vit import ViT
from .flash_pytorch import FLASHTransformer

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, input):
        return self.conv(input)

# class DoubleDFConv(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(DoubleDFConv, self).__init__()
#         self.conv = nn.Sequential(
#             DeformConv(in_ch, out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             DeformConv(out_ch, out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, input):
#         return self.conv(input)
# 可变形卷积
# class DeformConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.offset1=nn.Conv2d(in_channels,2*3*3,kernel_size=3,padding=1)
#         self.DConv1=DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
#
# # 前向传播，上一层的输出是下一层的输入
#     def forward(self, x):
#         # 获取偏置项1
#         offset1=self.offset1(x)
#         # 进行一次可变形卷积
#         d_x1=self.DConv1(x,offset1)
#         return d_x1

 
 
class FLASHUnet(nn.Module):
    def __init__(self,in_ch,out_ch,isStep=False):
        super(FLASHUnet, self).__init__() 
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1) 
        self.isStep = isStep

        self.flash_vit = FLASHTransformer(in_channel=256,path_size=1,emb_size=256, dim=256,depth=12,group_size=256,query_key_dim=128,img_size=16)
        # self.flash_vit2 = FLASHTransformer(in_channel=128,path_size=1,emb_size=128, dim=128,depth=12,group_size=256,query_key_dim=128,img_size=16)
        # self.flash_vit3 = FLASHTransformer(in_channel=256,path_size=1,emb_size=256, dim=256,depth=12,group_size=256,query_key_dim=128,img_size=8)
        # self.flash_vit4 = FLASHTransformer(in_channel=512,path_size=1,emb_size=512, dim=512,depth=12,group_size=256,query_key_dim=128,img_size=4)
        # self.flash_vit = FLASHTransformer(in_channel=512,path_size=1,emb_size=512, dim=512,depth=12,group_size=256,query_key_dim=128,img_size=4)
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        # c2: 得到32*32,128通道
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        # c3: 得到16*16,256通道
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        # c4: 得到8*8,512通道
        c4=self.conv4(self.flash_vit(p3))
        # p4=self.pool4(c4)
        # c5: 得到4*4,1024通道
        # c5=self.conv5(p4)
        # up_6= self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)
        # c6=self.conv6(merge6)
        # up_7=self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7=self.conv7(merge7)

        # # up_7: 得到16*16，256通道
        up_7=self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        #up_8: 得到32*32，128通道
        up_8=self.up8(c7)
        merge8=torch.cat([up_8,c2],dim=1)
        c8=self.conv8(merge8)
        #up_9: 得到64*64，64通道
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)     
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out


# class UnetTo3D(nn.Module):
#   def __init__(self,in_ch,out_ch):
#       super(UnetTo3D, self).__init__()
#       self.unet=VUnet(in_ch,out_ch)
#   def forward(self,x):
#     y = self.unet(x[:,:,0,:,:])
#     y=y.unsqueeze(2)
#     for i in range(1,x.shape[2],1):
#       # print(i)
#       # print(x[:,:,i,:,:].shape)
#       result = self.unet(x[:,:,i,:,:])
#       result = result.unsqueeze(2)
#       # print(result.shape)
#       y=torch.cat([y,result],dim=2)
#       # print(y.shape)
#     return y
    
      
if __name__ == '__main__':
    # da=torch.load('win/winWin2D/model_data/4/flash-unet/model/net.h5')
    # print(da)
    model = FLASHUnet(3, 1)
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    input = torch.randn(size=(2, 3,128, 128))
    out = model(input)
    print(out.shape)
    # vit=FLASHTransformer(in_channel=256, path_size=1, emb_size=256, dim=256, depth=12, group_size=256, query_key_dim=128,
    #                  img_size=8)
    # s=torch.randn(size=(2,256,8,8))
    # total = sum([param.nelement() for param in vit.parameters()])
    # print("Numberof parameter: % .2fM" % (total / 1024 ** 2))
    # print(vit(s).shape)