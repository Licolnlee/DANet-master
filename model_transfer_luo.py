import torch.utils.data.distributed
from torch.utils.mobile_optimizer import optimize_for_mobile
from models import highlight_detection, DANet

# pytorch环境中
# model_path = 'experiments0/checkpoint/unet/2022-09-13_4_0.0004_0_-1.0_1_sd2_rgb_340.pth'  # 模型的参数文件
model_path = 'experiments/checkpoint/unet/2022-07-15_16_0.0004_0_-1.0_1_sd2_rgb_500.pth'
ckpt = torch.load(model_path, map_location='cpu')
# net = highlight_detection.HighLightDetection('unet')
net = DANet.DANet(mode='unet')
net.load_state_dict(ckpt)

net.eval()  # 模型设为评估模式
# torch.save(net,'high_light_detection.pth')
# 1张3通道224*224的图片
#input_tensor = torch.rand(1, 3, 384, 512)  # 设定输入数据格式
input_tensor = torch.rand(1, 3, 512, 384)  # 设定输入数据格式

mobile = torch.jit.trace(net, input_tensor)  # 模型转化
optimized_traced_model = optimize_for_mobile(mobile)
optimized_traced_model._save_for_lite_interpreter("./mobile_2022-07-15_512.ptl")
#mobile.save('high_light_detection_luo.ptl')  # 保存文件
