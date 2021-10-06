import os
from os.path import join
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback
import datetime


class MLFlowLogger(Callback):
    def __init__(self, mlflow):
        super(MLFlowLogger, self).__init__()
        self.mlflow = mlflow

    def on_epoch_end(self, epoch, logs=None):
        self.mlflow.log_metric('train accuracy', logs['accuracy'])
        self.mlflow.log_metric('val accuracy', logs['val_accuracy'])
        self.mlflow.log_metric('train loss', logs['loss'])
        self.mlflow.log_metric('val_loss', logs['val_loss'])
        # self.mlflow.log_metric('lr', logs['lr'])
        self.mlflow.log_metric('train_dice', logs['dice'])
        self.mlflow.log_metric('val_dice', logs['val_dice'])
        self.mlflow.log_metric('train_iou', logs['iou'])
        self.mlflow.log_metric('val_iou', logs['val_iou'])

    def on_train_begin(self, logs=None):
        self.mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)


class MLFlowHandler:
    def __init__(self, model_name, run_name):
        self.mlflow_logger = MLFlowLogger(mlflow)
        self.mlflow = mlflow
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = model_name + "_" + str(datetime.datetime.now().date()) + "_" + str(
                datetime.datetime.now().time())
        self.model_name = model_name

    def start_run(self, args):
        self.mlflow.set_experiment(str(self.model_name))
        experiment = self.mlflow.get_experiment_by_name(str(self.model_name))
        ex_id = experiment.experiment_id
        self.mlflow.start_run(run_name=self.run_name, experiment_id=str(ex_id))
        command_line = "python train.py " + ' '.join([f"--{k} {v}" for k, v in args._get_kwargs()])
        self.mlflow.set_tag("command_line line", command_line)
        for k, v in args._get_kwargs():
            self.mlflow.log_param(k, v)

    def end_run(self, model_path=None):
        if model_path is not None:
            self.add_weight(model_path, )
        self.mlflow.end_run()

    def add_figure(self, figure, artifact_path):
        self.mlflow.log_figure(figure, artifact_path)

    def add_weight(self, weight_path, artifact_path=None):
        if artifact_path is None:
            weight_name = os.path.split(weight_path)[-1]
            artifact_path = join('models', weight_name)
        self.mlflow.log_artifact(weight_path, artifact_path)
