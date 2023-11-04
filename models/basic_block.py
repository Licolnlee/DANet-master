import torch
from torch import nn as nn
import torch.nn.functional as F


class Double_Branch_Net(nn.Module):

    def __init__(self, in_dim, out_dim=3):
        super(Double_Branch_Net, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                  nn.BatchNorm2d(48),
                                  nn.ReLU(inplace=True))
        # 构建原始分辨率分支
        ORB_RCAB = [RCAB(), RCAB(), RCAB(), RCAB(), RCAB()]
        self.ORB_RCAB = nn.Sequential(*ORB_RCAB)
        self.ORB_SFB = SFB(num=2)
        # 构建编码器模块
        self.encoder_list = nn.ModuleList()
        self.encoder_sfb = nn.ModuleList()
        # 构建解码器模块
        self.decoder_list = nn.ModuleList()
        self.decoder_sfb_1 = nn.ModuleList()
        self.decoder_sfb_2 = nn.ModuleList()
        for i in range(3):
            self.encoder_list.append(nn.Sequential(RCAB(depth_wise=True), RCAB(depth_wise=True)))
            self.encoder_sfb.append(SFB(num=1))

            self.decoder_list.append(nn.Sequential(RCAB(), RCAB()))
            if i == 0:
                self.decoder_sfb_1.append(SFB(num=3))
            else:
                self.decoder_sfb_1.append(SFB(num=4))
            self.decoder_sfb_2.append(SFB(num=1))

        self.out = nn.Conv2d(48, out_dim, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = self.conv(x)
        x_orb = self.ORB_RCAB(x)
        x = F.interpolate(x, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True, recompute_scale_factor=True)

        sfb = []
        for i in range(3):
            x = self.encoder_list[i](x)
            sfb.append(self.encoder_sfb[i]([x]))
            if i < 2:
                x = F.interpolate(x, scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True,
                                  recompute_scale_factor=True)
        # 第三层解码
        x = self.decoder_sfb_1[0]([F.interpolate(sfb[0], scale_factor=(0.25, 0.25), mode='bilinear', align_corners=True,
                                                 recompute_scale_factor=True),
                                   F.interpolate(sfb[1], scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True,
                                                 recompute_scale_factor=True),
                                   sfb[2]])
        x = self.decoder_list[0](x)
        x = F.interpolate(self.decoder_sfb_2[0]([x]), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # 第二层解码
        x = self.decoder_sfb_1[1]([F.interpolate(sfb[0], scale_factor=(0.5, 0.5), mode='bilinear', align_corners=True,
                                                 recompute_scale_factor=True),
                                   sfb[1],
                                   F.interpolate(sfb[2], scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                   x])
        x = self.decoder_list[1](x)
        x = F.interpolate(self.decoder_sfb_2[1]([x]), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        # 第一层解码
        x = self.decoder_sfb_1[2]([sfb[0],
                                   F.interpolate(sfb[1], scale_factor=(2, 2), mode='bilinear', align_corners=True),
                                   F.interpolate(sfb[2], scale_factor=(4, 4), mode='bilinear', align_corners=True),
                                   x])
        x = self.decoder_list[2](x)
        x = F.interpolate(self.decoder_sfb_2[2]([x]), scale_factor=(2, 2), mode='bilinear', align_corners=True)
        x = self.ORB_SFB([x_orb, x])
        return torch.sigmoid(self.out(x))


class RCAB(nn.Module):

    def __init__(self, depth_wise=False, in_dim=48, out_dim=48):
        super(RCAB, self).__init__()
        if depth_wise:
            self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 1, kernel_size=3, padding=1, padding_mode='replicate'),
                                       nn.Conv2d(1, 48, kernel_size=1), nn.ReLU(inplace=True),
                                       nn.Conv2d(48, 1, kernel_size=3, padding=1, padding_mode='replicate'),
                                       nn.Conv2d(1, 48, kernel_size=1), nn.ReLU(inplace=True))

        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                       nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(nn.Conv2d(48, 12, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(12, out_dim, kernel_size=1),
                                   )

    def forward(self, x):
        x = self.conv1(x)
        x_gap = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        sig = torch.sigmoid(self.conv2(x_gap))
        return x * sig + x


class SFB(nn.Module):

    def __init__(self, num=1, in_dim=48, out_dim=48):
        super(SFB, self).__init__()
        self.depth_wise_conv = nn.Sequential(
            nn.Conv2d(in_dim * num, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.Conv2d(1, 48, kernel_size=1), nn.ReLU(inplace=True))
        self.regular_conv = nn.Sequential(nn.Conv2d(48, 12, kernel_size=1), nn.ReLU(inplace=True),
                                          nn.Conv2d(12, out_dim, kernel_size=1))

    def forward(self, x_list):
        x = torch.cat(x_list, dim=1)
        x = self.depth_wise_conv(x)
        sig = torch.sigmoid(self.regular_conv(x))
        return x * sig + x


class ALM(nn.Module):
    def __init__(self, in_dim_1=3,in_dim_2=3):
        super(ALM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim_1, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim_2, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(48, 12, kernel_size=1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(12, 48, kernel_size=1),
                                   )
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim_1, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1, padding_mode='replicate'),
                                   )
        self.sfb = SFB(num=2)
        # self.out = nn.Sequential(nn.Conv2d(48, 3, kernel_size=3, stride=1, padding=1,padding_mode='replicate'),
        #                          nn.Sigmoid())

    def forward(self, bs, rs, rain):
        f_bs = self.conv1(bs)
        f_ti = torch.sigmoid(self.conv2(rs)) * self.conv3(rain)

        return self.sfb([f_bs, f_ti])


if __name__ == '__main__':
    model = ALM()
    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024 ** 2))
    # input = torch.randn(size=(2, 3, 64, 64))
    # out = model(input)
    # print(out.max(), out.min())
