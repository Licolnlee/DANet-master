import os
import warnings
import torch
from torch import nn as nn
import numpy as np
from torch.utils import data
import datetime
import tqdm
import pandas as pd

from dataloader import SD
from models import DANet, highlight_detection
from loss import Charbonnier, SSIM, Lab_loss
from config import args

torch.manual_seed(0)


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 2 ** 20


def train(args):
    # 创建需要的文件夹并指定gpu
    model_dir = os.path.join(args.model_dir, args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = os.path.join(args.log_dir, args.model)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 日志文件
    time = datetime.date.today().isoformat()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(device)
    save_name = time + "_" + str(args.batch) + '_' + str(args.lr) + '_' + str(args.wd) + '_' + str(
        args.alpha) + '_' + str(args.beta) + '_' + str(args.image_dir) + '_' + str(args.color)

    log_path = os.path.join(log_dir, save_name + ".csv")
    print("log_name: ", log_path)
    # 创建训练集
    train_data = SD(args.image_dir, mode='train')
    train_size = int(0.95 * len(train_data))
    val_size = len(train_data) - train_size
    # 重新划分数据集
    train_data, val_data = data.random_split(train_data, [train_size, val_size])
    # print(train_data)

    train_loader = data.DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    # 创建模型
    model = highlight_detection.HighLightDetection(args.model).to(device)
    print("AffNet:{0}M".format(count_parameters(model)))
    model.train()

    # 创建优化器及损失函数
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    charb = Charbonnier()
    ssim = SSIM()
    df = pd.DataFrame(columns=['iter', 'train_loss', 'val_loss'])
    sample_train = len(train_data) // args.batch
    for epoch in range(args.epochs):
        # common.adjust_learning_rate(opt, epoch, args)
        # print(opt.param_groups[0]['lr'])
        model.train()
        train_loss = .0
        for img, gt in tqdm.tqdm(train_loader):
            img = img.to(device)
            gt = gt.to(device)

            # 前向传播计算
            _, bs = model(img)

            loss1 = charb(bs, gt) + args.alpha * ssim(bs, gt)
            # loss1 = Lab_loss(bs, gt) + args.alpha * ssim(bs, gt)
            # print(loss1,loss2)
            optim.zero_grad()
            loss1.backward()
            optim.step()

            train_loss += loss1.item()

        print(f'epoch:{epoch},total loss:{train_loss / sample_train}')
        val_loss = val(model, val_loader, len(val_data), device)
        df = df.append(pd.Series(
            data=[epoch, train_loss / sample_train, val_loss],
            index=['iter', 'train_loss', 'val_loss']),
            ignore_index=True)
        df.to_csv(log_path, index=False)
        if (epoch + 1) % 10 == 0:
            # scheduler.step()
            torch.save(model.state_dict(), os.path.join(model_dir, save_name + '_{0}.pth'.format(epoch + 1)))


def val(model, val_loader, n, device):
    # 验证集
    model.eval()
    val_loss = .0
    charb = Charbonnier()
    ssim = SSIM()
    for img, gt in tqdm.tqdm(val_loader):
        img = img.to(device)
        gt = gt.to(device)

        # 前向传播计算
        _, bs = model(img)

        loss = charb(bs, gt) + args.alpha * ssim(bs, gt)
        # loss1 = Lab_loss(bs, gt) + args.alpha * ssim(bs, gt)

        val_loss += loss.item()
    return val_loss / n


if __name__ == "__main__":
    # torch.random.manual_seed(100)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
    train(args=args)
