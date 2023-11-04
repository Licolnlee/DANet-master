import os
import warnings
import torch
from torchvision import transforms, utils
from torch import nn as nn
import numpy as np
from torch.utils import data
import datetime
import tqdm
import pandas as pd
from PIL import Image

from dataloader import SD
from models import DANet
from loss import Charbonnier, SSIM
from config import args


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 2 ** 20


def test(args):
    # 创建需要的文件夹并指定gpu
    model_dir = os.path.join(args.model_dir, args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    result_dir = os.path.join(args.result_dir, args.model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # 指定设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(device)

    # 创建模型
    model = DANet.DANet(args.model).to(device)
    print("AffNet:{0}M".format(count_parameters(model)))
    model.eval()
    # 加载模型参数
    model_path = os.path.join(model_dir, r"2022-05-31_16_0.0004_0_-1.0_1_SHIQ_500.pth")
    print("load state dict from:", model_path)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    # torch.save(model, "mymodel1.pth")
    # 构建评价指标
    charb = Charbonnier()
    ssim = SSIM()
    # 创建测试集
    test_data = SD(args.test_dir, mode='test')
    sample = len(test_data)
    print('lengh', sample)
    trans = transforms.Compose([transforms.Resize(args.shape),
                                transforms.ToTensor()])
    # 循环测试集
    total_charb_bs = .0
    total_ssim_bs = .0

    total_charb_b = .0
    total_ssim_b = .0
    save_list = list(range(100))
    for i in range(sample):
        high_path = test_data.file_list[i]
        name = os.path.basename(high_path)
        low_path = high_path.replace('high_light', 'low_light')

        if args.test_dir in ['SHIQ', 'sd2']:
            low_path = low_path.replace('_light.png', '_gt.png')
        img = Image.open(high_path)
        gt = Image.open(low_path)
        print(trans(img).shape)
        # 转换为tensor
        img = torch.unsqueeze(trans(img), dim=0)[:, :3].to(device)
        gt = torch.unsqueeze(trans(gt), dim=0)[:, :3].to(device)
        print(img)

        # 前向传播计算
        rs, bs, b = model(img)

        # 计算量化指标
        charb_loss_bs = charb(bs, gt)  # 越低越好
        ssim_loss_bs = ssim(bs, gt)  # 越高越好

        charb_loss_b = charb(b, gt)  # 越低越好
        ssim_loss_b = ssim(b, gt)  # 越高越好
        if i in save_list:
            utils.save_image(torch.cat([img, rs.repeat(1, 3, 1, 1), bs, b, b - bs, gt], dim=0),
                             os.path.join(result_dir, f'{name}'))
        total_charb_bs += charb_loss_bs.item()
        total_ssim_bs += ssim_loss_bs.item()

        total_charb_b += charb_loss_b.item()
        total_ssim_b += ssim_loss_b.item()
    print(f'metric_bs:Charbonnier:{total_charb_bs / sample},SSIM{total_ssim_bs / sample}')
    print(f'metric_b:Charbonnier:{total_charb_b / sample},SSIM{total_ssim_b / sample}')


if __name__ == "__main__":
    torch.random.manual_seed(100)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    test(args=args)
