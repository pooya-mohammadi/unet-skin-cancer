"""
This module contains models for resent50
It's but an example. Modify it as you wish.
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


class UnetRes50:
    def __init__(self, img_w=256, img_h=256, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self):
        def conv_block(input, num_filters):
            x = Conv2D(num_filters, 3, padding="same")(input)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        def decoder_block(input, skip_features, num_filters):
            x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
            x = Concatenate()([x, skip_features])
            x = conv_block(x, num_filters)
            return x

        """ Input """
        inputs = Input(self.input_shape, name="input_1")

        """ Pre-trained ResNet50 Model """
        resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

        """ Encoder """
        s1 = resnet50.get_layer("input_1").output  ## (512 x 512)
        s2 = resnet50.get_layer("conv1_relu").output  ## (256 x 256)
        s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
        s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

        """ Bridge """
        b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)

        """ Decoder """
        d1 = decoder_block(b1, s4, 512)  ## (64 x 64)
        d2 = decoder_block(d1, s3, 256)  ## (128 x 128)
        d3 = decoder_block(d2, s2, 128)  ## (256 x 256)
        d4 = decoder_block(d3, s1, 64)  ## (512 x 512)

        """ Output """
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        model = Model(inputs, outputs, name="ResNet50_U-Net")
        model.summary()

        return model
