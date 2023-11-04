import os
import warnings
import torch
from torchvision import transforms, utils
from glob import glob
import numpy as np
from PIL import Image
import argparse

from models import DANet

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', default=False, help='use cpu only')
parser.add_argument("--gpu", type=int, help="gpu id",
                    dest="gpu", default=0)
parser.add_argument("--model", type=str, help="subnetwork",
                    dest="model", default='unet')
parser.add_argument("--shape", type=tuple, help="input shape",
                    dest="shape", default=(128, 128))
parser.add_argument("--input_dim", type=int, help="dimension of input image",
                    dest="input_dim", default=3)
parser.add_argument("--output_dim", type=int, help="dimension of highlight",
                    dest="output_dim", default=1)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='experiments/checkpoint')
args = parser.parse_args()


def infer(img):
    # 指定设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    # 创建模型
    model_dir = os.path.join(args.model_dir, args.model)
    model = DANet.DANet(args.model).to(device)
    model.eval()
    # 加载模型参数
    model_path = os.path.join(model_dir, r"2022-05-28_16_0.0004_0_-1.0_1_SHIQ_800.pth")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    # 前向传播计算
    rs, bs, x = model(img.to(device))
    return rs, bs, x


def preprocess(img_path):
    img = Image.open(img_path)
    old_size = img.size
    if max(old_size) < 0.5 * 256:
        ratio = 256 / max(old_size)
    elif min(old_size) > 2 * 256:
        ratio = 256 / min(old_size)
    else:
        ratio = 1
    new_size = int(old_size[1] * ratio // 8 * 8), int(old_size[0] * ratio // 8 * 8)
    # 转换为tensor
    trans = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor()])
    img = torch.unsqueeze(trans(img), dim=0)
    return img


def postprocess(img1):
    # 转换tensor为PIL 图像进行保存和显示
    img1 = img1.detach()
    img1 = torch.squeeze(img1, dim=0).permute(dims=[1, 2, 0])
    if img1.shape[-1] == 1:
        img1 = img1.repeat([1, 1, 3])
    img1 = img1.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()

    img1 = Image.fromarray(img1)
    return img1


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    img_path = r'data\a11_s_sd1.jpg'  # 图像路径
    img = preprocess(img_path)  # 图像预处理
    light, light_free, _ = infer(img)  # 高光检测
    light = postprocess(light)  # 转换tensor为PIL 图像进行保存和显示
    light_free = postprocess(light_free)

    light.show('light')
    light_free.show('light_free')
