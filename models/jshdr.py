import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch.nn.init as init


class ResBlock(nn.Module):
    def __init__(self, Channels, kSize=3):
        super(ResBlock, self).__init__()
        Ch = Channels
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(Ch, Ch, 3, padding=1, stride=1)

        self.conv3 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv4 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)

        self.conv5 = nn.Conv2d(Ch, Ch, 3, dilation=2, padding=2, stride=1)
        self.conv6 = nn.Conv2d(Ch, Ch, 3, dilation=4, padding=4, stride=1)

    def forward(self, x):
        x = x + self.relu(self.conv2(self.relu(self.conv1(x)))) + self.relu(
            self.conv4(self.relu(self.conv3(x)))) * 0.1 + self.relu(self.conv6(self.relu(self.conv5(x)))) * 0.1
        return x


class ResNet(nn.Module):
    def __init__(self, growRate0, nConvLayers=9, kSize=3):
        super(ResNet, self).__init__()
        G0 = growRate0
        C = nConvLayers

        self.convs = []

        self.res1 = ResBlock(G0)
        self.convs.append(self.res1)

        self.res2 = ResBlock(G0)
        self.convs.append(self.res2)

        self.res3 = ResBlock(G0)
        self.convs.append(self.res3)

        self.res4 = ResBlock(G0)
        self.convs.append(self.res4)

        self.res5 = ResBlock(G0)
        self.convs.append(self.res5)

        self.res6 = ResBlock(G0)
        self.convs.append(self.res6)

        self.res7 = ResBlock(G0)
        self.convs.append(self.res7)

        self.res8 = ResBlock(G0)
        self.convs.append(self.res8)

        self.res9 = ResBlock(G0)
        self.convs.append(self.res9)

        self.C = C

    def forward(self, x):
        feat_output = []

        for i in range(9):
            x = self.convs[i].forward(x)
            feat_output.append(x)
        return x, feat_output


class JSHDR_New(nn.Module):
    def __init__(self, args):
        super(JSHDR_New, self).__init__()
        # r = args.scale[0]
        G0 = 64
        kSize = args.RDNkSize

        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        self.encoder = nn.Conv2d(args.input_dim, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.updater = JSHDR_UNet(G0)
        # self.updater = ResNet(G0)
        self.mask_estimator1 = nn.Conv2d(G0, 8, kSize, padding=(kSize - 1) // 2, stride=1)
        self.mask_estimator2 = nn.Conv2d(8, args.output_dim, kSize, padding=(kSize - 1) // 2, stride=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x_F = self.updater(self.encoder(x))
        # x_F,_ = self.updater(self.encoder(x))
        x_mask = self.mask_estimator2(self.mask_estimator1(x_F))
        return torch.sigmoid(x_mask)


class JSHDR_UNet(nn.Module):

    def __init__(self, ch=64, dim=2):
        super(JSHDR_UNet, self).__init__()
        self.ch = ch

        self.down_fn = getattr(F, "max_pool{0}d".format(dim))

        self.deconv_fn = getattr(nn, "ConvTranspose{0}d".format(dim))

        self.encoder1 = self.get_conv_block(self.ch)
        self.encoder2 = self.get_conv_block(self.ch)
        self.encoder3 = self.get_conv_block(self.ch)
        self.encoder4 = self.get_conv_block(self.ch)

        self.decoder1 = self.get_conv_block(self.ch)
        self.decoder2 = self.get_conv_block(self.ch)
        self.decoder3 = self.get_conv_block(self.ch)

        self.up1 = self.deconv_fn(self.ch, self.ch, kernel_size=2, stride=2, padding=0)
        self.up2 = self.deconv_fn(self.ch, self.ch, kernel_size=2, stride=2, padding=0)
        self.up3 = self.deconv_fn(self.ch, self.ch, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        x = self.down_fn(enc1, 2, 2)
        enc2 = self.encoder2(x)
        x = self.down_fn(enc2, 2, 2)
        enc3 = self.encoder3(x)
        x = self.down_fn(enc3, 2, 2)

        x = self.encoder4(x)
        # Decoder path
        # x = torch.cat([enc3, self.up1(x)], dim=1)
        x = self.decoder1(enc3+self.up1(x))
        # x = torch.cat([enc2, self.up2(x)], dim=1)
        x = self.decoder2(enc2+ self.up2(x))
        # x = torch.cat([enc1, self.up3(x)], dim=1)
        x = self.decoder3(enc1+ self.up3(x))

        return x

    def get_conv_block(self,ch):
        return nn.Sequential(ResBlock(ch), ResBlock(ch))


if __name__ == '__main__':
    from config import args

    model = JSHDR_New(args)

    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024**2))
    torch.manual_seed(0)
    input = torch.randn(size=(2, 3, 128, 128))
    out = model(input)
    print(out.shape)
