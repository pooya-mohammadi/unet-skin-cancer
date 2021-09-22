import matplotlib.pyplot as plt
import numpy as np

def plot(testloader, Y_pred, index, history):
    def image_plot(testloader=testloader, Y_pred=Y_pred, index=index):
        plt.subplots_adjust(left=0.1,bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
        f = plt.figure(figsize=(10,10))
        ax1 = f.add_subplot(331)
        ax1.set_title(f"Input picture of index = {index}")
        ax1.imshow(testloader[0][index])
        ax2 = f.add_subplot(332)
        ax2.set_title(f"Mask picture of index = {index}")
        ax2.imshow(np.squeeze(testloader[1][index]))
        ax3 = f.add_subplot(333)
        ax3.set_title(f"Pred mask picture of index = {index}")
        ax3.imshow(np.squeeze(Y_pred[index]))
    
    def metric_plot(history = history):
        f = plt.figure(figsize=(12,6))
        ax1 = f.add_subplot(121)
        ax2 = f.add_subplot(122)
        ax1.plot(history.history['dice'])
        ax1.plot(history.history['val_dice'])
        ax1.set_xlabel('Dice coeficient')
        ax1.set_ylabel('Epoch')
        ax1.legend(['Train', 'Test'], loc='upper left')
        ax2.plot(history.history['iou'])
        ax2.plot(history.history['val_iou'])
        ax2.set_xlabel('IOU')
        ax2.set_ylabel('Epoch')
        ax2.legend(['Train', 'Test'], loc='upper left')
