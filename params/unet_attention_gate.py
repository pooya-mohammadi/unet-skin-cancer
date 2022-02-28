from argparse import ArgumentParser
from .main import main_args


def unetgate_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_attention_gate', help='Set it unet_attention_gate')
    return parser.parse_args()
