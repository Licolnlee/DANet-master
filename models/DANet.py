import torch
from torch import nn as nn
import numpy as np

from config import args
from .unet import UNet
from .basic_block import Double_Branch_Net, ALM
from .jshdr import JSHDR_New
# from train import count_parameters

# from .FLASHUnet import FLASHUnet

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 2 ** 20


class DANet(nn.Module):

    def __init__(self, mode='double'):
        super(DANet, self).__init__()
        if mode == 'double':
            self.DSN = Double_Branch_Net(in_dim=args.input_dim, out_dim=args.output_dim)
            self.RSN = Double_Branch_Net(in_dim=48, out_dim=args.input_dim)
        elif mode == 'jshdr':
            self.DSN = JSHDR_New(args)
            self.RSN = UNet(in_dim=48, out_dim=args.input_dim)
        else:
            self.DSN = UNet(in_dim=args.input_dim, out_dim=args.output_dim)
            self.RSN = UNet(in_dim=48, out_dim=args.input_dim)
        self.ALM = ALM(in_dim_1=args.input_dim, in_dim_2=args.output_dim)

    def forward(self, x):

        rs = self.DSN(x)
        # rs=torch.where(rs>0.2,rs,torch.tensor(0).float().to(rs.device))
        bs = x - rs
        # bs理论上应该控制在[0,1]之间
        # print(torch.max(rs), torch.min(rs))
        # bs = torch.clamp(bs, 0, 1)
        x = self.ALM(bs, rs, x)
        x = self.RSN(x)
        return rs, bs, x


if __name__ == '__main__':
    model = DANet(mode='unet')
    total = sum([param.nelement() for param in model.parameters()])
    print(model)
    print("Numberof parameter: % .3fM" % (total / 1024 ** 2))
    print("AffNet:{0}M".format(count_parameters(model)))
    data = torch.randn(size=(2, 3, 128, 128))
    data = data - data.min()
    data = data / data.max()
    print(data.min(), data.max())
    rs, bs, b = model(data)
    print(rs.shape)
    print(bs.shape)
    print(b.shape)
