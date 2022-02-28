from argparse import ArgumentParser
from .main import main_args


def unetConvDeconv_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_conv_deconv', help='model name. Set it unet_conv_deconv')
    return parser.parse_args()
