import sys
from datetime import datetime
from os.path import join
from models import load_model
from params import get_args
from data.data_loader import get_loader
from tensorflow.keras.optimizers import Adam
import mlflow
from utils.callbacks import get_callbacks
from utils.mlflow_handler import MLFlowHandler
from utils.plots import get_plots
from utils.metrics import dice, iou
from utils.loss import dice_loss, jaccard_loss, focal_tversky_loss
from utils.utils import get_gpu_grower

get_gpu_grower()


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")

    id_ = model_name + "_" + str(datetime.now().date()) + "_" + str(datetime.now().time())
    weight_path = join('weights', id_) + ".h5"
    mlflow_handler = MLFlowHandler(model_name=model_name,
                                   run_name=id_,
                                   run_ngrok=args.run_ngrok,
                                   mlflow_source=args.mlflow_source
                                   )
    mlflow_handler.start_run(args)
    # adjust paths for data_loader(add user path to folder path for each dataset)
    train_path = args.train_path + "/ISBI2016_ISIC_Part1_Training_Data"
    mask_train_path = args.mask_train_path + "/ISBI2016_ISIC_Part1_Training_GroundTruth"
    test_path = args.test_path + "/ISBI2016_ISIC_Part1_Test_Data"
    mask_test_path = args.mask_test_path + "/ISBI2016_ISIC_Part1_Test_GroundTruth"
    # Loading model
    train_loader, val_loader, test_loader = get_loader(train_path, mask_train_path,
                                                       test_path, mask_test_path,
                                                       batch_size=args.batch_size,
                                                       model_name=model_name,
                                                       cutmix_p=args.cutmix_p,
                                                       beta=args.cutmix_beta,
                                                       usual_aug_with_cutmix=args.usual_aug_with_cutmix,
                                                       )
    print("Loading Data is Done!")

    model = load_model(model_name=model_name)
    print("Loading Model is Done!")
    # training

    modelcheckpoint, _, _, estop_callback, lr_callback = get_callbacks(model_path=weight_path,
                                                                       early_stopping_p=5,
                                                                       mlflow=mlflow,
                                                                       args=args)
    if args.loss == 'dice_loss':
        loss = dice_loss
    elif args.loss == 'jaccard_loss':
        loss = jaccard_loss
    elif args.loss == 'focal_tversky_loss':
        loss = focal_tversky_loss
    else:
        raise Exception()
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=loss,
                  metrics=['acc', dice, iou])

    print("Training ....")
    model.fit(train_loader,
              batch_size=args.batch_size
              , epochs=args.epochs,
              validation_data=val_loader,
              callbacks=[estop_callback, modelcheckpoint, mlflow_handler.mlflow_logger])
    print("Training is done")
    get_plots(model, test_loader, args, mlflow_handler)
    mlflow_handler.end_run(weight_path)


if __name__ == '__main__':
    train()
