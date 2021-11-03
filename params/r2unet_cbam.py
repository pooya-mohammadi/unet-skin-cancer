from argparse import ArgumentParser
from .main import main_args

def r2unet_cbam_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='r2unet_cbam', help='Set it r2unet_cbam')
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