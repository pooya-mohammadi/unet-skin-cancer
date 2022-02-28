from argparse import ArgumentParser
from .main import main_args


def unetcbamgate_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_cbam_gate', help='model name. Set it unet_cbam_gate')
    return parser.parse_args()
