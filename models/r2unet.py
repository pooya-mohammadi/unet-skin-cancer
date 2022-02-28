from tensorflow.keras.layers import Conv2D, BatchNormalization, add
from keras_unet_collection.layer_utils import *
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


def RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr'):
    '''
    Recurrent convolutional layers with skip connection.
    
    RR_CONV(X, channel, kernel_size=3, stack_num=2, recur_num=2, activation='ReLU', batch_norm=False, name='rr')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
        
    '''

    activation_func = eval(activation)

    layer_skip = Conv2D(channel, 1, name='{}_conv'.format(name))(X)
    layer_main = layer_skip

    for i in range(stack_num):

        layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}'.format(name, i))(layer_main)

        if batch_norm:
            layer_res = BatchNormalization(name='{}_bn{}'.format(name, i))(layer_res)

        layer_res = activation_func(name='{}_activation{}'.format(name, i))(layer_res)

        for j in range(recur_num):

            layer_add = add([layer_res, layer_main], name='{}_add{}_{}'.format(name, i, j))

            layer_res = Conv2D(channel, kernel_size, padding='same', name='{}_conv{}_{}'.format(name, i, j))(layer_add)

            if batch_norm:
                layer_res = BatchNormalization(name='{}_bn{}_{}'.format(name, i, j))(layer_res)

            layer_res = activation_func(name='{}_activation{}_{}'.format(name, i, j))(layer_res)

        layer_main = layer_res

    out_layer = add([layer_main, layer_skip], name='{}_add{}'.format(name, i))

    return out_layer


def UNET_RR_left(X, channel, kernel_size=3,
                 stack_num=2, recur_num=2, activation='ReLU',
                 pool=True, batch_norm=False, name='left0'):
    '''
    The encoder block of R2U-Net.
    
    UNET_RR_left(X, channel, kernel_size=3, 
                 stack_num=2, recur_num=2, activation='ReLU', 
                 pool=True, batch_norm=False, name='left0')
    
    Input
    ----------
        X: input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    *downsampling is fixed to 2-by-2, e.g., reducing feature map sizes from 64-by-64 to 32-by-32
    '''
    pool_size = 2

    # maxpooling layer vs strided convolutional layers
    X = encode_layer(X, channel, pool_size, pool, activation=activation,
                     batch_norm=batch_norm, name='{}_encode'.format(name))

    # stack linear convolutional layers
    X = RR_CONV(X, channel, stack_num=stack_num, recur_num=recur_num,
                activation=activation, batch_norm=batch_norm, name=name)
    return X


def UNET_RR_right(X, X_list, channel, kernel_size=3,
                  stack_num=2, recur_num=2, activation='ReLU',
                  unpool=True, batch_norm=False, name='right0'):
    '''
    The decoder block of R2U-Net.
    
    UNET_RR_right(X, X_list, channel, kernel_size=3, 
                  stack_num=2, recur_num=2, activation='ReLU',
                  unpool=True, batch_norm=False, name='right0')
    
    Input
    ----------
        X: input tensor.
        X_list: a list of other tensors that connected to the input tensor.
        channel: number of convolution filters.
        kernel_size: size of 2-d convolution kernels.
        stack_num: number of stacked recurrent convolutional layers.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.
        batch_norm: True for batch normalization, False otherwise.
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor
    
    '''

    pool_size = 2

    X = decode_layer(X, channel, pool_size, unpool,
                     activation=activation, batch_norm=batch_norm, name='{}_decode'.format(name))

    # linear convolutional layers before concatenation
    X = CONV_stack(X, channel, kernel_size, stack_num=1, activation=activation,
                   batch_norm=batch_norm, name='{}_conv_before_concat'.format(name))

    # Tensor concatenation
    H = concatenate([X, ] + X_list, axis=-1, name='{}_concat'.format(name))

    # stacked linear convolutional layers after concatenation
    H = RR_CONV(H, channel, stack_num=stack_num, recur_num=recur_num,
                activation=activation, batch_norm=batch_norm, name=name)

    return H


def r2_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, recur_num=2,
                    activation='ReLU', batch_norm=False, pool=True, unpool=True, name='res_unet'):
    '''
    The base of Recurrent Residual (R2) U-Net.
    
    r2_unet_2d_base(input_tensor, filter_num, stack_num_down=2, stack_num_up=2, recur_num=2,
                    activation='ReLU', batch_norm=False, pool=True, unpool=True, name='res_unet')
    
    ----------
    Alom, M.Z., Hasan, M., Yakopcic, C., Taha, T.M. and Asari, V.K., 2018. Recurrent residual convolutional neural network 
    based on u-net (r2u-net) for medical image segmentation. arXiv preprint arXiv:1802.06955.
    
    Input
    ----------
        input_tensor: the input tensor of the base, e.g., `keras.layers.Inpyt((None, None, 3))`.
        filter_num: a list that defines the number of filters for each \
                    down- and upsampling levels. e.g., `[64, 128, 256, 512]`.
                    The depth is expected as `len(filter_num)`.
        stack_num_down: number of stacked recurrent convolutional layers per downsampling level/block.
        stack_num_down: number of stacked recurrent convolutional layers per upsampling level/block.
        recur_num: number of recurrent iterations.
        activation: one of the `tensorflow.keras.layers` or `keras_unet_collection.activations` interfaces, e.g., 'ReLU'.
        batch_norm: True for batch normalization.
        pool: True or 'max' for MaxPooling2D.
              'ave' for AveragePooling2D.
              False for strided conv + batch norm + activation.
        unpool: True or 'bilinear' for Upsampling2D with bilinear interpolation.
                'nearest' for Upsampling2D with nearest interpolation.
                False for Conv2DTranspose + batch norm + activation.                 
        name: prefix of the created keras layers.
        
    Output
    ----------
        X: output tensor.
    
    '''

    activation_func = eval(activation)

    X = input_tensor
    X_skip = []

    # downsampling blocks
    X = RR_CONV(X, filter_num[0], stack_num=stack_num_down, recur_num=recur_num,
                activation=activation, batch_norm=batch_norm, name='{}_down0'.format(name))
    X_skip.append(X)

    for i, f in enumerate(filter_num[1:]):
        X = UNET_RR_left(X, f, kernel_size=3, stack_num=stack_num_down, recur_num=recur_num,
                         activation=activation, pool=pool, batch_norm=batch_norm, name='{}_down{}'.format(name, i + 1))
        X_skip.append(X)

    # upsampling blocks
    X_skip = X_skip[:-1][::-1]
    for i, f in enumerate(filter_num[:-1][::-1]):
        X = UNET_RR_right(X, [X_skip[i], ], f, stack_num=stack_num_up, recur_num=recur_num,
                          activation=activation, unpool=unpool, batch_norm=batch_norm,
                          name='{}_up{}'.format(name, i + 1))

    return X


class R2Unet:
    def __init__(self, img_w=256, img_h=256, img_channels=3, filter_num=[64, 128, 256, 512], n_labels=1,
                 stack_num_down=2, stack_num_up=2, recur_num=2, activation='ReLU', output_activation='Sigmoid',
                 batch_norm=False, pool=True, unpool=True, name='r2_unet'):
        self.input_size = (img_w, img_h, img_channels)
        self.filter_num = filter_num
        self.n_labels = n_labels
        self.stack_num_down = stack_num_down
        self.stack_num_up = stack_num_up
        self.recur_num = recur_num
        self.activation = activation
        self.output_activation = output_activation
        self.batch_norm = batch_norm
        self.pool = pool
        self.unpool = unpool
        self.name = name

    def get_model(self):
        activation_func = eval(self.activation)

        IN = Input(self.input_size, name='{}_input'.format(self.name))

        # base
        X = r2_unet_2d_base(IN, self.filter_num,
                            stack_num_down=self.stack_num_down, stack_num_up=self.stack_num_up,
                            recur_num=self.recur_num,
                            activation=self.activation, batch_norm=self.batch_norm, pool=self.pool, unpool=self.unpool,
                            name=self.name)
        # output layer
        OUT = CONV_output(X, self.n_labels, kernel_size=1, activation=self.output_activation,
                          name='{}_output'.format(self.name))

        # functional API model
        model = Model(inputs=[IN], outputs=[OUT], name='{}_model'.format(self.name))

        return model
