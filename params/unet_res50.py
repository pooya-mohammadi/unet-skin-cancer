from .main import main_args


def unetRes50_args():
    parser = main_args()
    parser.add_argument('--model', type=str, default='unet_res50', help='model name. Set it unet_res50')
    return parser.parse_args()
