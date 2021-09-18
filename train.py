import sys
from models import load_model
from params import get_args
from data.data_loader import get_loader


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")
    train_loader, val_loader, test_loader = get_loader(args.dataset_dir)
    model = load_model(model_name=model_name, **args)

    # training


if __name__ == '__main__':
    train()
