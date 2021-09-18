from argparse import ArgumentParser


def resnet50_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50', help='model name. Set it resnet50')
    parser.add_argument('--epochs', type=int, default=10, help='define number of training epochs')
    return parser.parse_args()
