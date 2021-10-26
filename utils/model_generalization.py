from utils.combine_images import combine_images
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow
from data.data_loader import DataGenerator


def model_generalization(model: tensorflow.keras.models, test_loader: DataGenerator) -> plt.figure:
    """
    Explanation:
    This function get testloader and generates 4 images then put thm in one frame image
    then load model evaluate model on this image predict output and return related images
    """

    """
    
    Parameters
    ----------
    model
    test_loader

    Returns
    -------
    figure consist of 3 images x_test, y_test, y_pred
    """

    """
    This section design for generate 4 random images from test
    Because we read test according to batch_size=8 we can
    generate random 4 images with these 2 steps:
        1: generate random number(i) between 0 and len(test). in our case len(test) = 384 / 8 = 48
        2: we subtract this number in for loop until get zero then select first 4 photos from that batch
    for example if i = 20 then we select [x[160], y[160]], .....,[x[163], y[163]]
    """

    i = random.randint(1, len(test_loader))
    for x, y in test_loader:
        i = i - 1
        if (i == 0):
            x_img1, y_img1 = x[0], y[0]
            x_img2, y_img2 = x[1], y[1]
            x_img3, y_img3 = x[2], y[2]
            x_img4, y_img4 = x[3], y[3]
            break
    new_frame_x = combine_images(x_img1, x_img2, x_img3, x_img4)
    new_frame_y = combine_images(y_img1, y_img2, y_img3, y_img4)
    new_frame_x = np.expand_dims(new_frame_x, axis=0)
    new_frame_y = np.expand_dims(new_frame_y, axis=[0, 3])
    model.evaluate(x=new_frame_x, y=new_frame_y)
    y_predict = model.predict(new_frame_x)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.9, hspace=0.9)
    f = plt.figure(figsize=(15, 15))
    ax1 = f.add_subplot(331)
    ax1.imshow(new_frame_x[0])
    ax1.set_title(f"random 4 frame image")
    ax2 = f.add_subplot(332)
    ax2.imshow(np.squeeze(new_frame_y[0]))
    ax2.set_title(f"random 4 frame mask_image")
    ax3 = f.add_subplot(333)
    ax3.imshow(np.squeeze(y_predict[0]))
    ax2.set_title(f"random 4 frame predcited_image")

    return f
