from tensorflow.python.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, TensorBoard, CSVLogger
)
from deep_utils import log_print


def get_callbacks(model_path,
                  early_stopping_p,
                  tensorboard_dir,
                  csv_logger_dir,
                  reduce_lr_patience,
                  reduce_lr_factor,
                  plateau_min_lr,
                  save_weights_only=False,
                  save_best_only=True,
                  verbose=1,
                  logger=None):
    model_checkpoint = ModelCheckpoint(filepath=model_path,
                                       monitor='val_loss',
                                       save_best_only=save_best_only,
                                       mode='min',
                                       save_weights_only=save_weights_only,
                                       verbose=verbose
                                       )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=reduce_lr_factor,  # new_lr = lr * factor
                                  patience=reduce_lr_patience,  # number of epochs with no improvement
                                  min_lr=plateau_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=verbose
                                  )

    tensorboard = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True)
    csv_logger = CSVLogger(csv_logger_dir, append=True)
    # Unet callbacks
    # model_name = sys.argv[2]
    # args = get_args(model_name)

    # def get_lr_callback(batch_size=8, lr_start=args.lr_start, lr_min=args.lr_min):
    #     lr_max = 0.01 * batch_size
    #     lr_ramp_ep = 5
    #     lr_sus_ep = 0
    #     lr_decay = 0.8
    #
    #     def lrfn(epoch):
    #         if epoch < lr_ramp_ep:
    #             lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
    #
    #         elif epoch < lr_ramp_ep + lr_sus_ep:
    #             lr = lr_max
    #
    #         else:
    #             lr = (lr_max - lr_min) * lr_decay ** (epoch - lr_ramp_ep - lr_sus_ep) + lr_min
    #
    #         return lr
    #
    #     lr_callback = LearningRateScheduler(lrfn, verbose=False)
    #     return lr_callback

    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=early_stopping_p, mode='auto',
                               verbose=verbose)
    # batch_size = 8
    # lr_callback = get_lr_callback(batch_size)
    log_print(logger,
              "Successfully created following callbacks [model_checkpoint, reduce_lr, early_stop, csv_logger, tensorboard]")
    return model_checkpoint, reduce_lr, early_stop, csv_logger, tensorboard
