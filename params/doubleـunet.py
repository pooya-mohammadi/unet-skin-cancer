from .main import main_args


def doubleunet_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='double_unet', help='model name. Set it double_unet')
    return parser.parse_args()
