from argparse import ArgumentParser


def main_args():
    parser = ArgumentParser()

    # Training
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Choose the training optimizer. Default = adam')
    parser.add_argument('--batch_size', type=int, default=8, help='define size of each batch')
    parser.add_argument('--img_channel', type=int, default=3, help='number of channels image has')
    parser.add_argument('--transfer_learning_epochs', type=int, default=5,
                        help='Define the number of transfer learning epochs. Default = 5')
    parser.add_argument('--finetuning_epochs', type=int, default=10,
                        help='Define the number of fine tuning epochs. Default = 10')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Define the number of training epochs. Default = 10')

    # learning rate reduce on plateau
    parser.add_argument('--lr', type=float, default=0.01,
                        help=' learning rate. Default is 0.001')
    parser.add_argument('--min_lr', type=float, default=0.00001,
                        help='min_lr learning rate. Default is 0.001')

    # Callbacks
    parser.add_argument('--reduce_lr_patience', type=int, default=5,
                        help='Patience in learning rate schedule. Default is 5')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.1,
                        help='reduce lr factor. Default is 0.1')
    parser.add_argument('--early_stopping_p', type=int, default=5,
                        help='Patience for early stopping. Default is 5'
                        )

    # mlflow
    parser.add_argument('--mlflow-source', type=str, default='./mlruns', help='The mlflow directory')
    # parser.add_argument('--ngrok', dest='ngrok', action='store_true',
    #                     help="Run ngrok for colab!")
    # parser.add_argument('--no-ngrok', dest='ngrok', action='store_false',
    #                     help="Don't run ngrok for colab!")
    # parser.set_defaults(ngrok=False)

    # aug cut mix
    parser.add_argument('--cutmix_p', type=float, default=0.5,
                        help='probability to apply cutmix')
    parser.add_argument('--cutmix_beta', type=float, default=1,
                        help='beta variable of cutmix, default value is taken from the original paper')

    parser.add_argument('--usual_aug_with_cutmix', dest='usual_aug_with_cutmix', action='store_true',
                        help="Apply aug while applying cutmix")
    # Plots

    # augmentation
    parser.add_argument('--hair_aug_p', type=float, default=1,
                        help='probability to apply hair augmentation')
    parser.add_argument('--hair_rmv_p', dest='hair_rmv_p', type=float, default=0,
                        help='probability to apply hair augmentation')
    parser.add_argument('--random_rotate_p', type=float, default=0.5,
                        help='probability to apply random rotate')
    parser.add_argument('--p_horizontal_flip', type=float, default=0.5,
                        help='probability to apply p_horizontal_flip')
    parser.add_argument('--p_vertical_flip', type=float, default=0.5,
                        help='probability to apply p_vertical_flip')
    parser.add_argument('--p_center_crop', type=float, default=0.5,
                        help='probability to apply p_center_crop')
    parser.add_argument('--mosaic_p', type=float, default=0.5,
                        help='probability to apply mosaic_p')
    parser.add_argument("--hue_shift_limit", type=float, default=1,
                        help=" this is for defining hue_shift_limit, default is 0.5")
    parser.add_argument("--sat_shift_limit", type=float, default=0,
                        help=" this is for defining sat_shift_limit, default is 0.5")
    parser.add_argument("--contrast_limit", type=float, default=0.1,
                        help=" this is for defining contrast_limit, default is 0.5")
    parser.add_argument("--brightness_limit", type=float, default=0.1,
                        help=" this is for defining brightness_limit, default is 0.5")
    parser.add_argument("--hue_p", type=float, default=0.5, help=" this is for defining hue_p, default is 0.5")
    parser.add_argument("--contrast_p", type=float, default=0.5,
                        help=" this is for defining contrast_p, default is 0.5")
    parser.add_argument("--brightness_p", type=float, default=0.5,
                        help=" this is for defining brightness_p, default is 0.5")

    # Loss Function
    parser.add_argument('--loss', type=str, default='dice_loss',
                        help='Choose between binary_crossentropy and binary_focal_loss.')
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='Choose the value of label smoothing. 0 means no label smoothing.')
    parser.add_argument('--focal_loss_gamma', type=float, default=2,
                        help='Define the value for gamma parameter in focal loss.')
    parser.add_argument('--pos_weight', type=float, default=1,
                        help='Define the value of the weight for the positive class. Default 1')
    parser.add_argument('--neg_weight', type=float, default=1,
                        help='Define the value of the weight for the negative class. Default 1')

    # Arguments of path directions in dataloader
    parser.add_argument("--train_path", default="data/train", help='define train path images')
    parser.add_argument("--test_path", default="data/test", help='define test path images')
    parser.add_argument("--mask_train_path", default="data/mask_train", help='define mask of train path images')
    parser.add_argument("--mask_test_path", default="data/mask_test", help='define mask of test path images')
    parser.add_argument("--val_path", default="data/val", help='define mask of train path images')
    parser.add_argument("--mask_val_path", default="data/mask_val", help='define mask of test path images')

    # Other options
    parser.add_argument('--verbose', type=int, default=1,
                        help='Choose verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default = 1')
    parser.add_argument('--save_path', type=str, default='saved_models',
                        help="path to save model directory, default = saved_models")
    parser.add_argument("--dataset_name", type=str, default='ISIC_2016', help="dataset name, default=ISIC_2016")
    parser.add_argument("--seed", type=int, default=1234, help="This is seed! default is 1234")
    parser.add_argument("--img_size", type=tuple, default=(256, 256), help="img-size, default is (512, 512)")
    parser.add_argument('--save_path_name', type=str, default='',
                        help="specific name for saving models, default is None")
    return parser
