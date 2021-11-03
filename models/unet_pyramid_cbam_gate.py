from utils.attentionGate import *
from utils.cbam import *

from utils.attentionGate import *
from utils.cbam import *

class Pyramid_Cbam_Gate_Unet:
    def __init__(self, img_w=256, img_h=256, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self):
        kinit = 'glorot_normal'
        img_input = Input(shape=self.input_shape, name='input_scale1')
        scale_img_2 = AveragePooling2D(pool_size=(2, 2), name='input_scale2')(img_input)
        scale_img_3 = AveragePooling2D(pool_size=(2, 2), name='input_scale3')(scale_img_2)
        scale_img_4 = AveragePooling2D(pool_size=(2, 2), name='input_scale4')(scale_img_3)

        conv1 = UnetConv2D(img_input, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        input2 = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
        input2 = concatenate([input2, pool1], axis=3)
        conv2 = UnetConv2D(input2, 64, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        input3 = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
        input3 = concatenate([input3, pool2], axis=3)
        conv3 = UnetConv2D(input3, 128, is_batchnorm=True, name='conv3')
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        input4 = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
        input4 = concatenate([input4, pool3], axis=3)
        conv4 = UnetConv2D(input4, 64, is_batchnorm=True, name='conv4')
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        center = UnetConv2D(pool4, 512, is_batchnorm=True, name='center')

        conv1 = attach_attention_module(conv1, 'cbam_block')
        conv2 = attach_attention_module(conv2, 'cbam_block')
        conv3 = attach_attention_module(conv3, 'cbam_block')
        conv4 = attach_attention_module(conv4, 'cbam_block')

        g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
        attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
        up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                           kernel_initializer=kinit)(center), attn1], name='up1')

        g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
        attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
        up2 = concatenate([Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                           kernel_initializer=kinit)(up1), attn2], name='up2')

        g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
        attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
        up3 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                           kernel_initializer=kinit)(up2), attn3], name='up3')

        up4 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                           kernel_initializer=kinit)(up3), conv1], name='up4')

        # conv6 = UnetConv2D(up1, 256, is_batchnorm=True, name='conv6')
        # conv7 = UnetConv2D(up2, 128, is_batchnorm=True, name='conv7')
        # conv8 = UnetConv2D(up3, 64, is_batchnorm=True, name='conv8')
        conv9 = UnetConv2D(up4, 32, is_batchnorm=True, name='conv9')

        # out6 = Conv2D(1, (1, 1), activation='sigmoid', name='pred1')(conv6)
        # out7 = Conv2D(1, (1, 1), activation='sigmoid', name='pred2')(conv7)
        # out8 = Conv2D(1, (1, 1), activation='sigmoid', name='pred3')(conv8)
        out9 = Conv2D(1, (1, 1), activation='sigmoid', name='final')(conv9)

        model = Model(inputs=[img_input], outputs=[out9])
        return model
