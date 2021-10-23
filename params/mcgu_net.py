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
    # Arguments for Augmentation in dataloader
    parser.add_argument("--rotation_range", type=float, default=0.5, help='Range of rotation of images')
    parser.add_argument("--horizontal_flip", type=float, default=1.0, help='Whether flip horizontally or not')
    parser.add_argument("--vertical_flip", type=float, default=0.5, help='Whether flip horizontally or not')
    parser.add_argument("--center_crop", type=float, default=0.9, help='Whether flip horizontally or not')
    # Argument for learning_rate
    parser.add_argument("--lr_start", type=float, default=0.001, help='Learning_rate in first epochs')
    parser.add_argument("--lr_min", type=float, default=0.0001, help='Minimum learning_rate that exits')
    # Index for show rgb_mask_predmask of one image
    parser.add_argument("--index", type=int, default=125, help='Index for show rgb_mask_predmask of one image')

    return parser.parse_args()
