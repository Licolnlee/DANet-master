import torch
from torch import nn as nn

from config import args
from .unet import UNet
from .basic_block import Double_Branch_Net, ALM
from .jshdr import JSHDR_New


class HighLightDetection(nn.Module):

    def __init__(self, mode='double'):
        super(HighLightDetection, self).__init__()
        if mode == 'double':
            self.DSN = Double_Branch_Net(in_dim=args.input_dim, out_dim=args.output_dim)
        elif mode == 'jshdr':
            self.DSN = JSHDR_New(args)
        else:
            self.DSN = UNet(in_dim=args.input_dim, out_dim=args.output_dim)

    def forward(self, x):

        rs = self.DSN(x)
        # rs=torch.where(rs>0.2,rs,torch.tensor(0).float().to(rs.device))
        bs = x - rs

        return rs, bs


if __name__ == '__main__':
    model = HighLightDetection(mode='unet').cuda()
    total = sum([param.nelement() for param in model.parameters()])
    print("Numberof parameter: % .3fM" % (total / 1024 ** 2))
    data = torch.randn(size=(2, 3, 128, 128)).cuda()
    data = data - data.min()
    data = data / data.max()
    print(data.min(), data.max())
    rs, bs = model(data)
    print(rs.shape)
    print(bs.shape)

