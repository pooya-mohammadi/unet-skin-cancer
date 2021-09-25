from .unet_res50 import UnetRes50
from . unet_conv_deconv import UnetConvDeconv
ARGUMENTS = dict(unet_res50 = UnetRes50,
                 unet_conv_deconv = UnetConvDeconv,
              # other models
              )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
