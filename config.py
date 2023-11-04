import argparse

parser = argparse.ArgumentParser()

# 公共参数
parser.add_argument('--cpu', default=False, help='use cpu only')
parser.add_argument("--gpu", type=int, help="gpu id",
                    dest="gpu", default=0)
parser.add_argument("--model", type=str, help="subnetwork",
                    dest="model", default='DANet')
parser.add_argument("--shape", type=tuple, help="input shape",
                    dest="shape", default=(384,576))
parser.add_argument("--input_dim", type=int, help="dimension of input image",
                    dest="input_dim", default=3)
parser.add_argument("--output_dim", type=int, help="dimension of highlight",
                    dest="output_dim", default=1)
parser.add_argument("--image_dir", type=str, help="data folder with vols",
                    dest="image_dir", default=r"SHIQ")
# train时参数

parser.add_argument("--lr", type=float, help="learning rate",
                    dest="lr", default=0.001)
parser.add_argument("--wd", type=float, help="weight decay",
                    dest="wd", default=0)
parser.add_argument("--batch", type=int, help="batch size",
                    dest="batch", default=8)
parser.add_argument("--alpha", type=float, help="weight for the two loss",
                    dest="alpha", default=-0.15)
parser.add_argument("--beta", type=float, help="weight for the two networks",
                    dest="beta", default=1)
parser.add_argument("--epochs", type=int, help="num of epochs", default=500)
parser.add_argument("--model_dir", type=str, help="models folder",
                    dest="model_dir", default='experiments/checkpoint')
parser.add_argument("--log_dir", type=str, help="logs folder",
                    dest="log_dir", default='experiments/log')
parser.add_argument("--color", type=str, help="color space for loss function",
                    dest="color", default='rgb')
# test时参数
parser.add_argument("--test_dir", type=str, help="test data directory",
                    dest="test_dir", default='experiments/test')#r'SHIQ'
parser.add_argument("--result_dir", type=str, help="results folder",
                    dest="result_dir", default='experiments/result')  # experiments/result empire_part ghctc


# JSHDR_E模型的参数
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

args = parser.parse_args()
