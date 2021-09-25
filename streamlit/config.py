MODELS_TO_ADDR = {
    "unet": 'weights/unet/weights',
}

MODELS_TO_ARGS = {
    "unet-res50": {
        'input_shape': (None, None, 3)
    },
    "unet-conv-deconv": {
        'input_shape': (None, None, 3)
    },
}
