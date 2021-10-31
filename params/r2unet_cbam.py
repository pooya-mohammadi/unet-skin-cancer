from argparse import ArgumentParser


def r2unet_cbam_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='unet_attention_gate', help='Set it r2unet_cbam')
    parser.add_argument('--epochs', type=int, default= 20, help='define number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='define size of each batch')
    parser.add_argument('--lr', type=int, default=0.001, help='define learning_rate parameter')
    parser.add_argument('--loss', default="dice_loss", help='define loss function of model')
    #Arguments of path directions in dataloader
    parser.add_argument("--train_path", default = "./train" , help='define train path images')
    parser.add_argument("--test_path", default = "./test" , help='define test path images')
    parser.add_argument("--mask_train_path", default = "./masktrain", help='define mask of train path images')
    parser.add_argument("--mask_test_path", default = "./masktest" , help='define mask of test path images')
    #Arguments for Augmentation in dataloader
    parser.add_argument("--rotation_range", type=float, default=0.5, help='Range of rotation of images')
    parser.add_argument("--horizontal_flip", type=float, default=1.0, help='Whether flip horizontally or not')
    parser.add_argument("--vertical_flip", type=float, default=0.5, help='Whether flip horizontally or not')
    parser.add_argument("--center_crop", type=float, default=0.9, help='Whether flip horizontally or not')
    #Argument for learning_rate
    parser.add_argument("--lr_start", type=float, default=0.001, help='Learning_rate in first epochs')
    parser.add_argument("--lr_min", type=float, default=0.0001, help='Minimum learning_rate that exits')
    #Index for show rgb_mask_predmask of one image
    parser.add_argument("--index", type=int, default=125, help='Index for show rgb_mask_predmask of one image')

    return parser.parse_args()