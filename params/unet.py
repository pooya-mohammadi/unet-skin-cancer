from argparse import ArgumentParser


def UnetRes50():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='UNET', help='model name. Set it UNET')
    parser.add_argument('--epochs', type=int, default=10, help='define number of training epochs')
    parser.add_argument('--lr', type=int, default=0.001, help='define learning_rate parameter')
    parser.add_argument('--loss', type=str, default="binary_crossentropy", help='define loss function of model')
    parser.add_argument("--train_path", default="./train", help='define train path images')
    parser.add_argument("--test_path", default="./test", help='define test path images')
    parser.add_argument("--mask_train_path", default="./masktrain", help='define mask of train path images')
    parser.add_argument("--mask_test_path", default="./masktest", help='define mask of test path images')

    return parser.parse_args()
