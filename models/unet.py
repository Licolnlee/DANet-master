import torch
import torch.nn as nn
from torch.nn import functional as F


class UNet(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, dim=2, bn=True):
        super(UNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        self.down_fn = getattr(F, "max_pool{0}d".format(dim))

        self.deconv_fn = getattr(nn, "ConvTranspose{0}d".format(dim))

        self.bn = bn
        basic = 32  # 8 #32
        self.encoder1 = self.get_conv_block(in_dim=self.in_dim, mid_dim=basic, out_dim=2 * basic)
        self.encoder2 = self.get_conv_block(in_dim=2 * basic, mid_dim=2 * basic, out_dim=4 * basic)
        self.encoder3 = self.get_conv_block(in_dim=4 * basic, mid_dim=4 * basic, out_dim=8 * basic)
        self.encoder4 = self.get_conv_block(in_dim=8 * basic, mid_dim=8 * basic, out_dim=16 * basic)

        self.decoder1 = self.get_conv_block(in_dim=24 * basic, mid_dim=8 * basic, out_dim=8 * basic)
        self.decoder2 = self.get_conv_block(in_dim=12 * basic, mid_dim=4 * basic, out_dim=4 * basic)
        self.decoder3 = self.get_conv_block(in_dim=6 * basic, mid_dim=2 * basic, out_dim=2 * basic)

        self.up1 = self.deconv_fn(16 * basic, 16 * basic, kernel_size=2, stride=2, padding=0)
        self.up2 = self.deconv_fn(8 * basic, 8 * basic, kernel_size=2, stride=2, padding=0)
        self.up3 = self.deconv_fn(4 * basic, 4 * basic, kernel_size=2, stride=2, padding=0)

        self.classifier = nn.Sequential(self.conv_fn(2 * basic, self.out_dim, kernel_size=1, stride=1, padding=0),
                                        nn.Sigmoid())

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        x = self.down_fn(enc1, 2, 2)
        enc2 = self.encoder2(x)
        x = self.down_fn(enc2, 2, 2)
        enc3 = self.encoder3(x)
        x = self.down_fn(enc3, 2, 2)

        x = self.encoder4(x)
        # Deconder path
        x = torch.cat([enc3, self.up1(x)], dim=1)
        x = self.decoder1(x)
        x = torch.cat([enc2, self.up2(x)], dim=1)
        x = self.decoder2(x)
        x = torch.cat([enc1, self.up3(x)], dim=1)
        x = self.decoder3(x)
        # Classify layer
        x = self.classifier(x)
        return x

    def get_conv_block(self, in_dim, mid_dim, out_dim):
        encoder = []
        encoder.append(self.conv_fn(in_dim, mid_dim, kernel_size=3, stride=1, padding=1))
        if self.bn:
            encoder.append(self.bn_fn(mid_dim))
        encoder.append(nn.ReLU(inplace=True))

        encoder.append(self.conv_fn(mid_dim, out_dim, kernel_size=3, stride=1, padding=1))
        if self.bn:
            encoder.append(self.bn_fn(out_dim))
        encoder.append(nn.ReLU(inplace=True))
        return nn.Sequential(*encoder)


if __name__ == '__main__':
    model = UNet(3, 3, dim=2)
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .2fM" % (total / 1024 ** 2))
    input = torch.randn(size=(2, 3, 64, 64))
    out = model(input)
    print(out.shape)
