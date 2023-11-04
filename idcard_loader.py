import os
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
import cv2
from PIL import Image


class IDCard(data.Dataset):

    def __init__(self, root, ratio=3):
        self.root = root
        self.ratio = int(ratio)
        self.gt_root = os.path.join(root, 'gt')
        self.image_root = os.path.join(root, 'image')

        self.gt_list = os.listdir(self.gt_root)
        self.gt_list.sort()
        self.image_list = os.listdir(self.image_root)
        self.image_list.sort(key=lambda x: (len(x), x))

        assert len(self.gt_list) * self.ratio == len(self.image_list)
        self.trans = transforms.Compose([transforms.Resize(size=(192,288)),
                                  transforms.ToTensor()])

    def __getitem__(self, item):
        img_path = os.path.join(self.image_root, self.image_list[item])
        img = Image.open(img_path)
        #print(img_path)
        gt_path = os.path.join(self.gt_root, self.gt_list[item // self.ratio])
        gt = Image.open(gt_path)
        #print(gt_path)
        return self.trans(img).float(), self.trans(gt).float()

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    train_data = IDCard(root='idcard_train')
    img, gt = train_data[503]
    img = np.asarray(img)
    gt = np.asarray(gt)
    print(img.shape)
    #cv2.imshow('img', np.transpose(img[::-1], [1, 2, 0]))
    #cv2.imshow('gt', np.transpose(gt[::-1], [1, 2, 0]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
