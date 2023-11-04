import onnx
import onnxruntime
import numpy as np
import os
import warnings
import torch
from PIL.Image import Resampling
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
from PIL import Image
import cv2

if __name__ == '__main__':
    onnx_model = onnxruntime.InferenceSession('test4.onnx')
    # # 创建测试集
    # test_data = SD(args.test_dir, mode='test')
    # sample = len(test_data)
    # print('lengh', sample)
    # trans = transforms.Compose([transforms.Resize(args.shape),
    #                             transforms.ToTensor()])
    # input_data = np.array((1, 3, 256, 256))
    #
    # input_data = input_data.astype(np.float32)

    image = Image.open("SHIQ/SpecularHighlightTest/high_light/nohighlight-14001.png")
    image = image.resize((256, 256), Resampling.LANCZOS)
    cv2.imwrite("test6.png", np.array(image))
    image = (np.array(image)).astype(np.float32) / 255;
    input_data = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)

    print(input_data.shape)
    print(input_data)

    output = onnx_model.run(None, {'input': input_data})

    output_data = (output[0]*255).astype(np.uint8)
    output_data1 = (output[1]*255).astype(np.uint8)
    output_data2 = (output[2]*255).astype(np.uint8)

    print(type(output_data))
    print(output_data.transpose(0, 2, 3, 1)[0].shape)

    cv2.imwrite("test5.png", (output_data.transpose(0, 2, 3, 1)[0]))
    cv2.imwrite("test7.png", (output_data1.transpose(0, 2, 3, 1)[0]))
    cv2.imwrite("test8.png", (output_data2.transpose(0, 2, 3, 1)[0]))

    # im = Image.fromarray((output_data.transpose(0, 2, 3, 1)[0]*255).astype(np.uint8))

    # print(im.shape)
    # im.save("test5.png")

    print(output_data)
# 我们可以使用异常处理的方法进行检验
# try:
#     # 当我们的模型不可用时，将会报出异常
#     onnx.checker.check_model("test4.onnx")
# except onnx.checker.ValidationError as e:
#     print("The model is invalid: %s" % e)
# else:
#     # 模型可用时，将不会报出异常，并会输出“The model is valid!”
#     print("The model is valid!")
