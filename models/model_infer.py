import torch
from torch import nn as nn
import torch.nn.functional as  F
from torchvision import transforms
from PIL import Image

from config import args
from .unet import UNet
from .basic_block import Double_Branch_Net, ALM
from .jshdr import JSHDR_New
from .FLASHUnet import FLASHUnet


class DANet_UP(nn.Module):

    def __init__(self, mode='double'):
        super(DANet_UP, self).__init__()
        if mode == 'double':
            self.DSN = Double_Branch_Net(in_dim=args.input_dim, out_dim=args.output_dim)
            self.RSN = Double_Branch_Net(in_dim=48, out_dim=args.input_dim)
        elif mode == 'jshdr':
            self.DSN = JSHDR_New(args)
            self.RSN = UNet(in_dim=48, out_dim=args.input_dim)
        elif mode == 'flash':
            self.DSN = FLASHUnet(args.input_dim, args.output_dim)
            self.RSN = UNet(in_dim=48, out_dim=args.input_dim)
        else:
            self.DSN = UNet(in_dim=args.input_dim, out_dim=args.output_dim)
            self.RSN = UNet(in_dim=48, out_dim=args.input_dim)
        self.ALM = ALM(in_dim_1=args.input_dim, in_dim_2=args.output_dim)

    def forward(self, path):
        # x输入图像
        x, size = self.preprocess(path)
        rs = self.DSN(x)
        # rs=torch.where(rs>0.2,rs,torch.tensor(0).float().to(rs.device))
        bs = x - rs
        # bs理论上应该控制在[0,1]之间
        # print(torch.max(rs), torch.min(rs))
        # bs = torch.clamp(bs, 0, 1)
        x = self.ALM(bs, rs, x)
        x = self.RSN(x)
        # 后处理上采样
        rs_up = F.interpolate(rs, size=size[::-1], mode='bicubic', align_corners=True)
        bs_up = F.interpolate(bs, size=size[::-1], mode='bicubic', align_corners=True)

        # x_up=F.interpolate(bs, size=size, mode='bicubic', align_corners=True)

        bs_up = self.postprocess(bs_up)
        rs_up = self.postprocess(rs_up)
        return rs_up, bs_up

    def preprocess(self, path):
        img = Image.open(path)
        old_size = img.size
        # 转换为tensor
        trans = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
        img = torch.unsqueeze(trans(img), dim=0)

        return img, old_size

    def postprocess(self, x):
        # 转换tensor为PIL 图像进行保存和显示
        img1 = x.detach()
        img1 = torch.squeeze(img1, dim=0).permute(dims=[1, 2, 0])
        if img1.shape[-1] == 1:
            img1 = img1.repeat([1, 1, 3])
        img1 = img1.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()

        img1 = Image.fromarray(img1)
        return img1


if __name__ == '__main__':
    import os

    path = 'data/5.jpg'
    net = DANet_UP(mode='unet')
    net.eval()
    # 加载模型参数
    model_path = os.path.join('experiments/checkpoint/unet', r"2022-05-28_16_0.0004_0_-1.0_1_SHIQ_800.pth")
    ckpt = torch.load(model_path, map_location='cpu')
    net.load_state_dict(ckpt)

    light, light_free = net(path)
    light.show()
    light_free.show()
