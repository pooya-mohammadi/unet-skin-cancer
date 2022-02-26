import os.path
import sys
from datetime import datetime
from os.path import join
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.optimizers import Adam
from utils.callbacks import get_callbacks
from utils.metrics import dice_score, iou
from utils.loss import dice_loss, jaccard_loss, focal_tversky_loss, get_loss
from utils.mlflow_handler import MLFlowHandler
from utils.plots import evaluation
from utils.utils import get_gpu_grower
from deep_utils import get_logger, save_params, mkdir_incremental, log_print, tf_set_seed

get_gpu_grower()


def train():
    model_name = sys.argv[2]
    args = get_args(model_name)
    tf_set_seed(args.seed)
    dataset_name = args.dataset_name
    save_path = os.path.join(args.save_path, dataset_name, model_name, args.save_path_name)
    save_path = mkdir_incremental(save_path)
    logger = get_logger("SKIN-CANCER", log_path=os.path.join(save_path, "skin_cancer.log"))
    save_params(join(save_path, "params.txt"), args, logger=logger)
    log_print(logger, f"Chosen Model: {model_name}")
    log_print(logger, f" Arguments: {args}")
    model_path = join(save_path, "model")

    mlflow_handler = MLFlowHandler(model_name=model_name,
                                   run_name=f"{dataset_name}_{model_name}_{args.save_path_name}",
                                   mlflow_source=args.mlflow_source
                                   )
    mlflow_handler.start_run(args)

    # Loading model
    train_loader, val_loader, test_loader = get_loader(args.train_path,
                                                       args.mask_train_path,
                                                       args.test_path,
                                                       args.mask_test_path,
                                                       batch_size=args.batch_size,
                                                       model_name=model_name,
                                                       cutmix_p=args.cutmix_p,
                                                       beta=args.cutmix_beta,
                                                       usual_aug_with_cutmix=args.usual_aug_with_cutmix,
                                                       hair_aug_p=args.hair_aug_p,
                                                       hair_rmv_p=args.hair_rmv_p,
                                                       img_channel=args.img_channel,
                                                       img_size=args.img_size,
                                                       random_rotate_p=args.random_rotate_p,
                                                       p_horizontal_flip=args.p_horizontal_flip,
                                                       p_vertical_flip=args.p_vertical_flip,
                                                       p_center_crop=args.p_center_crop,
                                                       p_mosaic=args.mosaic_p,
                                                       seed=args.seed,
                                                       logger=logger
                                                       )

    model = load_model(model_name=model_name, logger=logger, img_w=args.img_size[0], img_h=args.img_size[1])
    # training

    model_checkpoint, reduce_lr, early_stop, csv_logger, tensorboard = get_callbacks(model_path=model_path,
                                                                                     early_stopping_p=args.early_stopping_p,
                                                                                     tensorboard_dir=join(save_path,
                                                                                                          "tensorboard_train"),
                                                                                     csv_logger_dir=join(save_path,
                                                                                                         "csv_logger_train.csv"),
                                                                                     save_weights_only=False,
                                                                                     reduce_lr_factor=args.reduce_lr_factor,
                                                                                     plateau_min_lr=args.min_lr,
                                                                                     verbose=args.verbose,
                                                                                     reduce_lr_patience=args.reduce_lr_patience)
    loss = get_loss(args.loss, logger)
    model.compile(
        optimizer=Adam(learning_rate=args.lr),
        loss=loss,
        metrics=[dice_score, iou, dice_loss, jaccard_loss, focal_tversky_loss]
    )

    log_print(logger, f"Start train for {save_path}")
    history = model.fit(train_loader,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_data=val_loader,
                        callbacks=[model_checkpoint, reduce_lr, early_stop, csv_logger, tensorboard,
                                   mlflow_handler.mlflow_logger])
    log_print(logger, f"Training is done model is saved in {save_path}")

    # Loading checkpoint model
    # model = keras_load_model(save_path)
    # model.load_weights(model_path)
    evaluation(
        model,
        test_loader,
        mlflow_handler,
        save_path=save_path,
        logger=logger
    )
    # mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()
