from argparse import ArgumentParser
from .main import main_args


def mcgUnet_args() -> ArgumentParser():
    """

    Returns
    This function return argparser for give parameters in train.py
    -------

    """

    """
    These are parameters such as batch_size, epoch, learning_rate,
    augmentation parameters and....
    """
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_conv_deconv', help='model name. Set it unet_conv_deconv')

    return parser.parse_args()
