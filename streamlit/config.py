MODELS_TO_ADDR = {
    "unet_res50": 'weights/unet_res50/unet_model.h5',
    "unet_conv_deconv": 'weights/unet_conv_deconv/weights',
}

MODELS_TO_ARGS = {
    "unet_res50": {
        'input_shape': (None, None, 3)
    },
    "unet_conv_deconv": {
        'input_shape': (None, None, 3)
    },
}
