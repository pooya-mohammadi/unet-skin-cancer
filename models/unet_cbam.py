from utils.attentionGate import *
from utils.cbam import *

class CbamUnet:
    def __init__(self, img_w=256, img_h=256, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self):
        kinit = 'glorot_normal'
        inputs = Input(shape=self.input_shape)
        conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetConv2D(pool1, 64, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = UnetConv2D(pool2, 128, is_batchnorm=True, name='conv3')
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = UnetConv2D(pool3, 256, is_batchnorm=True, name='conv4')
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)

        cbam1 = attach_attention_module(conv1, 'cbam_block')
        cbam2 = attach_attention_module(conv2, 'cbam_block')
        cbam3 = attach_attention_module(conv3, 'cbam_block')
        cbam4 = attach_attention_module(conv4, 'cbam_block')

        up6 = concatenate(
            [Conv2DTranspose(256, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv5), cbam4],
            axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), cbam3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

        up8 = concatenate(
            [Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv7), cbam2],
            axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)

        up9 = concatenate(
            [Conv2DTranspose(32, (2, 2), strides=(2, 2), kernel_initializer=kinit, padding='same')(conv8), cbam1],
            axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        return model