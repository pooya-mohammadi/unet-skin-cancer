from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def get_callbacks(model_path, early_stopping_p, save_weights_only=True, plateau_min_lr=0.0001, **kwargs):
    checkpoint = ModelCheckpoint(filepath=model_path,
                                 monitor='val_loss',
                                 save_best_only=True,
                                 mode='min',
                                 save_weights_only=save_weights_only,
                                 )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.9,  # new_lr = lr * factor
                                  patience=10,  # number of epochs with no improvment
                                  min_lr=plateau_min_lr,  # lower bound on the learning rate
                                  mode='min',
                                  verbose=1
                                  )
    early_stopping = EarlyStopping(monitor="val_loss", patience=early_stopping_p, verbose=1)
    return checkpoint, reduce_lr, early_stopping
