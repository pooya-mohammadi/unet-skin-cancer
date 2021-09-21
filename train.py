import sys

from tensorflow.keras import callbacks
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from utils.callbacks import *

def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")
    train_loader, val_loader, test_loader = get_loader(args.train_path, args.test_path, 
                                                       args.mask_train_path, args.mask_test_path,
                                                       args.rotation_range, args.width_shift_range,
                                                       args.height_shift_range, args.shear_range,
                                                       args.zoom_range, args.fill_mode,
                                                       args.horizontal_flip)
    model = load_model(model_name=model_name, **args)

    # training
    model.compile(optimizer = Adam(learning_rate = args.lr), loss = args.loss, 
              metrics = ['accuracy'])
        
    batch_size = 8
    train_step = int((900*0.8)/batch_size)
    val_step = int((900*0.2)/batch_size)

    history = model.fit_generator(train_loader, steps_per_epoch=train_step, epochs=args.epoch,
                              validation_data=val_loader, validation_steps=val_step,
                              callbacks = [get_callbacks])

    #Save model
    model.save_weights("unet_model.h5")
    #Load model
    model.load_weights("unet_model.h5")
if __name__ == '__main__':
    train()
