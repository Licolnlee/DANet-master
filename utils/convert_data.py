import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

root = r'G:\workspace\jshdr\dataset\SD1'
save_root = r'G:\workspace\jshdr\src\dataset\SHIQ'
for mode in ['train', 'val']:
    data_path = os.path.join(root, mode)
    for idx, img in enumerate(os.listdir(data_path)):
        img_path = os.path.join(data_path, img)
        print(img_path)
        img = Image.open(img_path)
        img = np.array(img)
        # 提取数据
        nolight = Image.fromarray(img[:, :512, :3])
        highlight = Image.fromarray(img[:, 512:1024, :3])
        light = img[:, 1024:, :3]
        if mode == 'train':
            save_path = os.path.join(save_root, 'SpecularHighlight')
        else:
            save_path = os.path.join(save_root, 'SpecularHighlightTest')
        nolight_path=os.path.join(save_path,'nohighlight')
        highlight_path=os.path.join(save_path,'highlight')
        #保存图片
        nolight.save(os.path.join(nolight_path,'nohighlight-'+str(idx+1)+'.png'))
        highlight.save(os.path.join(highlight_path, 'nohighlight-' + str(idx + 1) + '.png'))
