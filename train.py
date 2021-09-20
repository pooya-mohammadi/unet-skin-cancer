import sys
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")
    train_loader, val_loader, test_loader = get_loader(args.train_path, args.test_path, 
                                                       args.mask_train_path, args.mask_test_path)
    model = load_model(model_name=model_name, **args)

    # training
    model.compile(optimizer = Adam(learning_rate = args.lr), loss = args.loss, 
              metrics = ['accuracy'])
    train_step = int((900*0.8)/8)
    val_step = int((900*0.2)/8)

    history = model.fit_generator(train_loader, steps_per_epoch=train_step, epochs=args.epoch,
                              validation_data=val_loader, validation_steps=val_step)

if __name__ == '__main__':
    train()
