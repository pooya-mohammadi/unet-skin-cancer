# Copyright 2021 The AI-Medic\Skin_Cancer Authors. All Rights Reserved.

"""Deep auto-encoder-decoder network for medical image segmentation
with state of the art results on skin lesion segmentation. This method applies bidirectional convolutional LSTM layers
in U-net structure to non-linearly encode both semantic and high-resolution
information with non-linearly technique. Furthermore, it applies densely connected
convolution layers to include collective knowledge in representation and
boost convergence rate with batch normalization layers.


In the table below you can see the result of MCG_UNet on Isic2016 dataset:

+--------------+-----------------------+----------------------+--------------+--------------+
| Model Name   |   Isic-2016 test Dice |   Isic-2016 test iou |   Params (M) |  Attention   |
+==============+=======================+======================+==============+==============+
| MCG_UNet     |                 81.35 |                69.84 |          1.7 | Convlstm     |
+--------------+-----------------------+----------------------+--------------+--------------+

Reference:
        - [Multi-level Context Gating of Embedded Collective Knowledge for Medical Image Segmentation](https://arxiv.org/pdf/2003.05056v1.pdf)
        - [BCDU_Net Code](https://github.com/rezazad68/BCDU-Net)
        - [Paperwithcode](https://paperswithcode.com/paper/multi-level-context-gating-of-embedded)

This code has been implemented in python language using Keras library with
tensorflow backend and tested in windows OS, though should be compatible with related environment
"""

from utils.attentionGate import *
from utils.sqeezeexcitation import *
import numpy as np
from typing import Tuple, Union


class MCGUNET:
    def __init__(self, img_w: int = 256, img_h: int = 256, channels: int = 3, **kwargs) -> None:
        self.input_shape = (img_w, img_h, channels)
        """
        Parameters:
            input_shape: shape tuple, in "channels_last" format;
            it should have exactly 3 inputs channels, and width and
            height should be no smaller than 32.
            - image_size = (256,256,3)
        """

    def get_model(self) -> tensorflow.keras.Model:
        """
        This method returns a Keras image segmentation model.
        Returns
        A Tensorflow.keras.Model` instance.
        """

        # kinit = 'glorot_normal'
        kinit = 'he_normal'
        input_img = Input(self.input_shape)
        N = self.input_shape[0]
        # Block1
        """
        UnetConv2d is the function that execute 2 Conv2D.
        if is_batchnorm is true it executes batch_normalization after each Conv2D layer
        """
        conv1 = UnetConv2D(input_img, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1)

        # Block2
        conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2)

        # Block3
        x = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')

        # D1
        drop4_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        # D2
        conv4_2 = UnetConv2D(drop4_1, 128, is_batchnorm=True, name='conv4')
        conv4_2 = Dropout(0.5)(conv4_2)
        # D3
        merge_dense = concatenate([conv4_2, drop4_1], axis=3)
        conv4_3 = UnetConv2D(merge_dense, 128, is_batchnorm=True, name='conv5')
        drop4_3 = Dropout(0.5)(conv4_3)

        up6 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same', kernel_initializer=kinit)(drop4_3)
        up6 = squeezeexcite(up6, ratio=16)
        up6 = BatchNormalization(axis=3)(up6)
        up6 = Activation('relu')(up6)

        x1 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 64))(conv2)
        x2 = Reshape(target_shape=(1, np.int32(N / 2), np.int32(N / 2), 64))(up6)
        merge6 = concatenate([x1, x2], axis=1)
        merge6 = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                            kernel_initializer='he_normal')(merge6)

        conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge6)
        conv6 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv6)
        conv6 = squeezeexcite(conv6, ratio=16)

        up7 = Conv2DTranspose(32, kernel_size=2, strides=2, padding='same', kernel_initializer=kinit)(conv6)
        up7 = squeezeexcite(up7, ratio=16)
        up7 = BatchNormalization(axis=3)(up7)
        up7 = Activation('relu')(up7)

        x1 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 32))(conv1)
        x2 = Reshape(target_shape=(1, np.int32(N), np.int32(N), 32))(up7)
        merge7 = concatenate([x1, x2], axis=1)
        merge7 = ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=False, go_backwards=True,
                            kernel_initializer='he_normal')(merge7)

        conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kinit)(merge7)
        conv7 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv7)
        conv7 = squeezeexcite(conv7, ratio=16)

        conv8 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer=kinit)(conv7)
        conv9 = Conv2D(1, 1, activation='sigmoid')(conv8)

        model = Model(inputs=[input_img], outputs=[conv9])

        return model
