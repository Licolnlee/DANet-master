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
from glob import glob
from PIL import Image,ImageFilter
import cv2

from dataloader import SD
from models import DANet
from loss import Charbonnier, SSIM
from config import args

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 2 ** 20


def infer(args):
    # 创建需要的文件夹并指定gpu
    model_dir = os.path.join(args.model_dir, args.model)
    result_dir = os.path.join('infer_id_local_loss', args.model)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # 指定设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(device)

    # 创建模型
    model = DANet.DANet(args.model).to(device)
    print("AffNet:{0}M".format(count_parameters(model)))
    # print(model)
    model.eval()
    # 加载模型参数
    model_path = os.path.join(model_dir, r"2022-10-26_8_0.0004_0_-1.0_1_idcard_train_rgb_200.pth")
    print("load state dict from:", model_path)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    # 创建测试集
    img_list = glob('data1/*.png') + glob('data1/*.jpg') + glob('data1/*.jepg')
    sample = len(img_list)

    # 循环测试集
    for i in range(sample):
        high_path = img_list[i]
        name = os.path.basename(high_path)
        img = Image.open(high_path)
        #img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        old_size = img.size
        new_size = old_size[1] // 8 * 8, old_size[0] // 8 * 8
        print(new_size)
        # 转换为tensor
        trans = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor()])
        img = torch.unsqueeze(trans(img), dim=0).to(device)

        # 前向传播计算
        rs, bs, b = model(img)
        # print('-'*20)
        # print(rs.max(),rs.min())
        # print(b.max(),b.min())
        utils.save_image(torch.cat([img, rs.repeat(1, 3, 1, 1), bs, b, b - bs], dim=0),
                         os.path.join(result_dir, f'{name}'))


def show(image, window_name):
    cv2.namedWindow(window_name, 0)
    cv2.imshow(window_name, image)
    # 0任意键终止窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_card_location(path):
    image = cv2.imread(path)
    show(image, "image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show(gray, "gray")
    blur = cv2.medianBlur(gray, 7)
    show(blur, "blur")
    threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    show(threshold, "threshold")
    canny = cv2.Canny(threshold, 100, 150)
    show(canny, "canny")
    kernel = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=5)
    show(dilate, "dilate")
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_copy = image.copy()
    res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
    show(res, "res")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    image_copy = image.copy()
    res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 20)
    show(res, "contours")
    epsilon = 0.02 * cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, epsilon, True)
    n = []
    for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
        n.append((x, y))
    n = sorted(n)
    sort_point = []
    n_point1 = n[:2]
    n_point1.sort(key=lambda x: x[1])
    sort_point.extend(n_point1)
    n_point2 = n[2:4]
    n_point2.sort(key=lambda x: x[1])
    n_point2.reverse()
    sort_point.extend(n_point2)
    p1 = np.array(sort_point, dtype=np.float32)
    h = sort_point[1][1] - sort_point[0][1]
    w = sort_point[2][0] - sort_point[1][0]
    pts2 = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(p1, pts2)
    dst = cv2.warpPerspective(image, M, (w, h))
    # print(dst.shape)
    show(dst, "dst")
    if w < h:
        dst = np.rot90(dst)
    resize = cv2.resize(dst, (384, 576), interpolation=cv2.INTER_AREA)

    show(resize, "resize")
    exit()

def get_preprocessed_image():
    pass


if __name__ == "__main__":
    # torch.random.manual_seed(100)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    # file_root = 'data1/front'
    # file = 'e55f0651-12c1-4d88-81fc-df0d45bbc3ef.jpg'
    # file_path = os.path.join(file_root, file)
    # get_card_location(file_path)

    infer(args=args)
