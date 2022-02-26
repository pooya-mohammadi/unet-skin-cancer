import os

import matplotlib.pyplot as plt
import numpy as np
from deep_utils import log_print

from .mlflow_handler import MLFlowHandler
from utils.model_generalization import model_generalization


def image_plot(testloader, model, logger=None):
    figures = []
    for x, y in testloader:
        log_print(logger, "predicting for plots!")
        y_pred = model.predict(x)
        for i in range(8):
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
            f = plt.figure(figsize=(10, 10))
            ax1 = f.add_subplot(331)
            ax1.set_title(f"Input picture of index = {i}")
            ax1.imshow(x[i])
            ax2 = f.add_subplot(332)
            ax2.imshow(np.squeeze(y[i]))
            ax2.set_title(f"Mask picture of index = {i}")
            ax3 = f.add_subplot(333)
            ax3.imshow(np.squeeze(y_pred[i]))
            ax3.set_title(f"Predicted mask picture of index = {i}")
            figures.append(f)
        break
    return figures


def evaluation(model, test_loader, mlflow_handler: MLFlowHandler, save_path, logger=None):
    # Metrics: Test: Loss, Acc, Dice, Iou
    print('Evaluation')

    test_score = model.evaluate(test_loader)  # test data
    log_print(logger,
              f'Test: Loss= {test_score[0]}, Dice-Score: {test_score[1]}, IoU: {test_score[2]}, Dice_loss: {test_score[3]}, jaccard_loss: {test_score[4]}, focal_tversky_loss: {test_score[5]}')

    # Metrics: Confusion Matrix

    figures = image_plot(test_loader, model, logger=logger)
    for i in range(len(figures)):
        mlflow_handler.add_figure(figures[i], f'images/rgb_test_mask_samples{i}.png')
        figures[i].savefig(os.path.join(save_path, f'rgb_test_mask_samples{i}.png'))

    figure = model_generalization(model, test_loader, logger=logger)
    mlflow_handler.add_figure(figure, f'images/model_generalization_samples.png')
    figure.savefig(os.path.join(save_path, 'model_generalization_samples.png'))
    log_print(logger, "Successfully Saved figures!")
