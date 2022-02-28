from argparse import ArgumentParser
from .main import main_args


def unetcbam_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_cbam', help='model name. Set it unet_cbam')
    return parser.parse_args()
