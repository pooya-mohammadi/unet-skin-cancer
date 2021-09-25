from .unet_res50 import unetRes50_args
from . unet_conv_deconv import unetConvDeconv_args
ARGUMENTS = dict(unet_res50 = unetRes50_args,
                 unet_conv_deconv = unetConvDeconv_args,
              # other models
              )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
