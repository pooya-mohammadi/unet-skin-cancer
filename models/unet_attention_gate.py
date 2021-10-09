from utils.attentionGate import *

class GateUnet:
    def __init__(self, img_w=256, img_h=256, channels=3, **kwargs):
        self.input_shape = (img_w, img_h, channels)

    def get_model(self):
        kinit = 'glorot_normal'
        inputs = Input(shape=self.input_shape)
        conv1 = UnetConv2D(inputs, 32, is_batchnorm=True, name='conv1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = UnetConv2D(pool1, 32, is_batchnorm=True, name='conv2')
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = UnetConv2D(pool2, 64, is_batchnorm=True, name='conv3')
        # conv3 = Dropout(0.2,name='drop_conv3')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = UnetConv2D(pool3, 64, is_batchnorm=True, name='conv4')
        # conv4 = Dropout(0.2, name='drop_conv4')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        center = UnetConv2D(pool4, 128, is_batchnorm=True, name='center')

        # Use CBAM with attention gate together
        # conv4 = attach_attention_module(conv4, 'cbam_block')
        # conv3 = attach_attention_module(conv3, 'cbam_block')
        # conv2 = attach_attention_module(conv2, 'cbam_block')

        g1 = UnetGatingSignal(center, is_batchnorm=True, name='g1')
        attn1 = AttnGatingBlock(conv4, g1, 128, '_1')
        up1 = concatenate([Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu',
                                           kernel_initializer=kinit)(center), attn1], name='up1')

        g2 = UnetGatingSignal(up1, is_batchnorm=True, name='g2')
        attn2 = AttnGatingBlock(conv3, g2, 64, '_2')
        up2 = concatenate(
            [Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up1),
             attn2], name='up2')

        g3 = UnetGatingSignal(up1, is_batchnorm=True, name='g3')
        attn3 = AttnGatingBlock(conv2, g3, 32, '_3')
        up3 = concatenate(
            [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up2),
             attn3], name='up3')

        up4 = concatenate(
            [Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu', kernel_initializer=kinit)(up3),
             conv1], name='up4')
        out = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=kinit, name='final')(up4)

        model = Model(inputs=[inputs], outputs=[out])
        return model
