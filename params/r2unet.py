from .main import main_args


def r2unet_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='r2unet', help='Set it r2unet')
    return parser.parse_args()
