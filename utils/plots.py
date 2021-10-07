import matplotlib.pyplot as plt
import numpy as np
from .mlflow_handler import MLFlowHandler
import random


def image_plot(testloader, Y_pred):
    figures = []
    for i in range(5):
        index = random.randint(0, 379)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
        f = plt.figure(figsize=(10, 10))
        ax1 = f.add_subplot(331)
        ax1.set_title(f"Input picture of index = {index}")
        ax1.imshow(testloader[0][index])
        ax2 = f.add_subplot(332)
        ax2.set_title(f"Mask picture of index = {index}")
        ax2.imshow(np.squeeze(testloader[1][index]))
        ax3 = f.add_subplot(333)
        ax3.set_title(f"Pred mask picture of index = {index}")
        ax3.imshow(np.squeeze(Y_pred[index]))
        figures.append(f)
    return figures


def get_plots(model, test_loader, args, mlflow_handler: MLFlowHandler):
    # Metrics: Test: Loss, Acc, Dice, Iou
    print('Evaluation')
    Y_pred = model.predict(test_loader)

    test_score = model.evaluate(test_loader)  # test data
    print(f'Test: Loss= {test_score[0]}, Accuracy: {test_score[1]}')
    print(f'Test: Dice= {test_score[2]}, Iou: {test_score[3]}')



    # Metrics: Confusion Matrix

    figures = image_plot(test_loader, Y_pred)
    for i in range(len(figures)):
        mlflow_handler.add_figure(figures[i], f'images/rgb_test_mask_samples{i}.png')
