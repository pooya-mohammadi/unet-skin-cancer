from argparse import ArgumentParser


def main_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        help='Model name. Default = resnet50')

    # Training
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Choose the training optimizer. Default = adam')
    parser.add_argument('--transfer_learning_epochs', type=int, default=5,
                        help='Define the number of transfer learning epochs. Default = 5')
    parser.add_argument('--finetuning_epochs', type=int, default=10,
                        help='Define the number of fine tuning epochs. Default = 10')

    # learning rate reduce on plateau
    parser.add_argument('--min_lr', type=float, default=0.001,
                        help='min_lr learnig rate. Default is 0.001')
    parser.add_argument('--lr_patience', type=int, default=10,
                        help='Patience in learning rate schedule. Default is 10')

    # ngrok
    parser.add_argument('--mlflow-source', type=str, default='./mlruns',
                        help='The mlflow direcotry')
    parser.add_argument('--run-ngrok', dest='run_ngrok', default=True, action='store_true',
                        help="Run ngrok for colab!")
    parser.add_argument('--no-run-ngrok', dest='run_ngrok', action='store_false',
                        help="Don't run ngrok for colab!")

    # Plots
    # roc curve
    parser.add_argument('--plot_roc', dest='plot_roc', action='store_true',
                        help="Plot the roc curve.")
    parser.add_argument('--no-plot_roc', dest='plot_roc', action='store_false',
                        help="Don't plot the roc curve.")
    parser.set_defaults(plot_roc=True)
    # precision-recall curve
    parser.add_argument('--plot_pr', dest='plot_pr', action='store_true',
                        help="Plot the precision-recall curve.")
    parser.add_argument('--no-plot_pr', dest='plot_pr', action='store_false',
                        help="Don't plot the precision-recall curve.")
    parser.set_defaults(plot_pr=True)

    # Loss Function
    parser.add_argument('--loss', type=str, default='binary_crossentropy',
                        help='Choose between binary_crossentropy and binary_focal_loss.')
    parser.add_argument('--label_smoothing', type=float, default=0,
                        help='Choose the value of label smoothing. 0 means no label smoothing.')
    parser.add_argument('--focal_loss_gamma', type=float, default=2,
                        help='Define the value for gamma parameter in focal loss.')
    parser.add_argument('--pos_weight', type=float, default=1,
                        help='Define the value of the weight for the positive class.')
    parser.add_argument('--neg_weight', type=float, default=1,
                        help='Define the value of the weight for the negative class.')

    # Other options
    parser.add_argument('--verbose', type=int, default=1,
                        help='Choose verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. Default = 1')

    return parser
