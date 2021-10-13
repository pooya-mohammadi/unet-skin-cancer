import tensorflow.keras.backend as K
from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow
from tensorflow.keras.layers import *


def squeezeexcite(x: tensorflow.keras.layers, ratio=16) -> tensorflow.keras.layers:
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
