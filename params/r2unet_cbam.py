from argparse import ArgumentParser
from .main import main_args

def r2unet_cbam_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='r2unet_cbam', help='Set it r2unet_cbam')
    return parser.parse_args()