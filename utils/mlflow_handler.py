import os
from os.path import join
import mlflow
import mlflow.tensorflow
from tensorflow.keras.callbacks import Callback
import datetime


class MLFlowLogger(Callback):
    def __init__(self, mlflow: mlflow):
        super(MLFlowLogger, self).__init__()
        self.mlflow = mlflow

    def on_epoch_end(self, epoch, logs=None):
        self.mlflow.log_metric('train dice', logs['dice'], step=epoch)
        self.mlflow.log_metric('val dice', logs['val_dice'], step=epoch)
        self.mlflow.log_metric('train iou', logs['iou'], step=epoch)
        self.mlflow.log_metric('val iou', logs['val_iou'], step=epoch)
        self.mlflow.log_metric('train loss', logs['loss'], step=epoch)
        self.mlflow.log_metric('val_loss', logs['val_loss'], step=epoch)

    def on_train_begin(self, logs=None):
        self.mlflow.log_param('optimizer_name', type(self.model.optimizer).__name__)


class MLFlowHandler:
    def __init__(self, model_name, run_name, mlflow_source='./mlruns', run_ngrok=True):
        self.mlflow = mlflow
        self.run_ngrok = run_ngrok
        self.mlflow_source = mlflow_source
        self.mlflow.set_tracking_uri(mlflow_source)
        self.mlflow_logger = MLFlowLogger(mlflow)
        if run_name is not None:
            self.run_name = run_name
        else:
            self.run_name = model_name + "_" + str(datetime.datetime.now().date()) + "_" + str(
                datetime.datetime.now().time())
        self.model_name = model_name

    @staticmethod
    def colab_ngrok(mlflow_source):
        from pyngrok import ngrok

        # run tracking UI in the background
        os.system(f"cd {os.path.split(mlflow_source)[0]} && mlflow ui --port 5000 &")

        ngrok.kill()

        # Setting the authtoken (optional)
        # Get your authtoken from https://dashboard.ngrok.com/auth
        NGROK_AUTH_TOKEN = ""
        ngrok.set_auth_token(NGROK_AUTH_TOKEN)

        # Open an HTTPs tunnel on port 5000 for http://localhost:5000
        ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
        print("MLflow Tracking UI:", ngrok_tunnel.public_url)

    def start_run(self, args):
        self.mlflow.set_experiment(str(self.model_name))
        experiment = self.mlflow.get_experiment_by_name(str(self.model_name))
        ex_id = experiment.experiment_id
        self.mlflow.start_run(run_name=self.run_name, experiment_id=str(ex_id))
        command_line = "python train.py " + ' '.join([f"--{k} {v}" for k, v in args._get_kwargs()])
        self.mlflow.set_tag("command_line line", command_line)
        for k, v in args._get_kwargs():
            self.mlflow.log_param(k, v)
        if self.run_ngrok:
            self.colab_ngrok(self.mlflow_source)

    def end_run(self, model_path=None):
        if model_path is not None:
            self.add_weight(model_path, )
        self.mlflow.end_run()

    def add_figure(self, figure, artifact_path):
        self.mlflow.log_figure(figure, artifact_path)

    def add_report(self, report, artifact_path):
        self.mlflow.log_text(report, artifact_path)

    def add_weight(self, weight_path, artifact_path=None):
        if artifact_path is None:
            weight_name = os.path.split(weight_path)[-1]
            artifact_path = join('models', weight_name)
        self.mlflow.log_artifact(weight_path, artifact_path)
