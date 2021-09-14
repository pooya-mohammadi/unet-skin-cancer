"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model


class Resnet50:
    def __init__(self, img_w=200, img_h=200, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self) -> Model:

        # get the pretrained model
        base_model = tf.keras.applications.ResNet50(input_shape=self.input_shape,
                                                    include_top=False,
                                                    weights='imagenet')
        base_model.trainable = False
        model = Sequential()
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model
