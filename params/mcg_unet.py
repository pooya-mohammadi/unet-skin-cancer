from argparse import ArgumentParser
from .main import main_args


def mcg_unet_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='mcg_unet', help='Set it mcg_unet')
    return parser.parse_args()
