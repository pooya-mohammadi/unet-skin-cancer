import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow
from tensorflow.keras.layers import *


def squeezeexcite(x: tensorflow.keras.layers, ratio: int = 16) -> tensorflow.keras.layers:
    """
    Explanations:
    SE blocks are lightweight gating mechanism in the channel-wise relationships.
    networks are able to learn now how to understand the importance of each feature
    map in the stack of all the feature maps extracted after a convolution operation
    and recalibrates that output to reflect that importance before passing the volume to the next layer
    """

    """
    Code link :
    https://github.com/rezazad68/BCDU-Net/blob/033a2e5768d87e04e39deacb4746ecf088518f63/Skin%20Lesion%20Segmentation/models.py#L187"""

    """

    Parameters
    ----------
    x : input layer of SE block
    ratio : parameter for making first dense layer

    Returns
    -------
    y : output of SE block
    """

    nb_channel = K.int_shape(x)[-1]
    y = GlobalAveragePooling2D()(x)
    y = Dense(nb_channel // ratio, activation='relu')(y)
    y = Dense(nb_channel, activation='sigmoid')(y)
    y = Multiply()([x, y])
    return y
