MODELS_TO_ADDR = {
    "unet_res50": 'saved_models/unet_res50/unet_model.h5',
    "unet_conv_deconv": 'saved_models/unet_conv_deconv/saved_models',
}

MODELS_TO_ARGS = {
    "unet_res50": {
        'input_shape': (None, None, 3)
    },
    "unet_conv_deconv": {
        'input_shape': (None, None, 3)
    },
}
