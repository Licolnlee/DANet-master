import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils import data as data
from PIL import Image
from glob import glob



class SD(data.Dataset):

    def __init__(self, root='SHIQ', mode='train'):
        self.root = root
        self.mode = mode
        self.file_list = self.get_file_list()

        self.trans = transforms.Compose([transforms.Resize(size=(512, 512)),
                                         transforms.ToTensor(),
                                         # transforms.Lambda(lambda x:x/255),
                                         ])

    def get_file_list(self):
        if self.mode == 'train':
            return glob(os.path.join(self.root, 'SpecularHighlight', 'high_light') + '/*.png')
        else:
            return glob(os.path.join(self.root, 'SpecularHighlightTest', 'high_light') + '/*.png')

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        high_path = self.file_list[item]
        low_path = high_path.replace('high_light', 'low_light')

        img = Image.open(high_path)
        gt = Image.open(low_path)

        img = self.trans(img).float()
        gt = self.trans(gt).float()
        return img, gt


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    data_set = SD(mode='test')
    print(len(data_set))

    img, gt = data_set[0]
    img = np.asarray(img) * 255
    print(img.shape)
    img = np.int_(np.transpose(img, axes=(1, 2, 0)))
    plt.imshow(img)
    plt.show()
    gt = np.int_(np.transpose(gt * 255, axes=(1, 2, 0)))
    print(gt.shape)
    plt.imshow(gt)
    plt.show()
    # print(img.shape)
    # loader=data.DataLoader(data_set,batch_size=2,shuffle=True,num_workers=0)
