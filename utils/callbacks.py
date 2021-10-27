from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from params import get_args
import sys


def get_callbacks(model_path, early_stopping_p, save_weights_only=True, plateau_min_lr=0.0001, **kwargs):
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=save_weights_only,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.8,  # new_lr = lr * factor
                                  patience=5,  # number of epochs with no improvment
                                  min_lr=plateau_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)

    # Unet callbacks
    model_name = sys.argv[2]
    args = get_args(model_name)

    def get_lr_callback(batch_size=8, lr_start=args.lr_start, lr_min=args.lr_min):
        lr_max = 0.01 * batch_size
        lr_ramp_ep = 5
        lr_sus_ep = 0
        lr_decay = 0.8

        def lrfn(epoch):
            if epoch < lr_ramp_ep:
                lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start

            elif epoch < lr_ramp_ep + lr_sus_ep:
                lr = lr_max

            else:
                lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min

            return lr

        lr_callback = LearningRateScheduler(lrfn, verbose=False)
        return lr_callback

    estop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=12, mode='auto')
    batch_size = 8
    lr_callback = get_lr_callback(batch_size)
    return checkpoint, reduce_lr, early_stopping, estop_callback, lr_callback
