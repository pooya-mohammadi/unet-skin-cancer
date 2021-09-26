MODELS_TO_ADDR = {
    "unet-res50": 'weights/unet_res50/weights',
    "unet-conv-deconv" : 'weights/unet_conv_deconv/weights',
}

MODELS_TO_ARGS = {
    "unet-res50": {
        'input_shape': (None, None, 3)
    },
    "unet-conv-deconv": {
        'input_shape': (None, None, 3)
    },
}
