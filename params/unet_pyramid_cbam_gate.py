from argparse import ArgumentParser
from .main import main_args


def unetcbamgate_pyramid_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_pyramid_cbam_gate', help='Set it unet_pyramid_cbam_gate')
    return parser.parse_args()
