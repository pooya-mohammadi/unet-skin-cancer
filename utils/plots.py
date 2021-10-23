import matplotlib.pyplot as plt
import numpy as np
from .mlflow_handler import MLFlowHandler
import random
from utils.model_generalization import model_generalization


def image_plot(testloader, model):
    figures = []
    for x, y in testloader:
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


def get_plots(model, test_loader, args, mlflow_handler: MLFlowHandler):
    # Metrics: Test: Loss, Acc, Dice, Iou
    print('Evaluation')

    test_score = model.evaluate(test_loader)  # test data
    print(f'Test: Loss= {test_score[0]}, Accuracy: {test_score[1]}')
    print(f'Test: Dice= {test_score[2]}, Iou: {test_score[3]}')

    # Metrics: Confusion Matrix

    figures = image_plot(test_loader, model)
    for i in range(len(figures)):
        mlflow_handler.add_figure(figures[i], f'images/rgb_test_mask_samples{i}.png')

    figure = model_generalization(model, test_loader)
    mlflow_handler.add_figure(figure, f'images/model_generalization_samples.png')
