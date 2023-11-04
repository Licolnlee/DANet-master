import os
import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from config import args
from models import DANet


def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params / 2 ** 20


# torch --> onnx

if __name__ == "__main__":
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
    model = model.eval().to(device)
    # 加载模型参数
    model_path = os.path.join(model_dir, r"2022-05-31_16_0.0004_0_-1.0_1_SHIQ_500.pth")
    print("load state dict from:", model_path)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)

    # batch_size = 1  # 批处理大小
    # input_shape = (3, 320, 320)

    x = torch.randn(1, 3, 256, 256).to(device)

    # x = torch.randn(batch_size, *input_shape)  # 生成张量
    # x = x.to(device)
    export_onnx_file = "test4.onnx"  # 目的ONNX文件名
    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=11,
                      # do_constant_folding=False,
                      # verbose=False,  # 是否执行常量折叠优化
                      input_names=["input"],  # 输入名
                      output_names=["output"],  # 输出名
                      # dynamic_axes={"input": {0: "batch_size"},  # 批处理变量
                      #               "output": {0: "batch_size"}}
                      )
    # onnx_model = onnx.load("test.onnx")
    # onnx.checker.check_model(onnx_model)
    # print("1")
