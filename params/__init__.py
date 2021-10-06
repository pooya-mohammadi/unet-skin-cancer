from .unet_res50 import unetRes50_args
from .unet_conv_deconv import unetConvDeconv_args
from .doubleunet import doubleunet_args

ARGUMENTS = dict(unet_res50=unetRes50_args,
                 unet_conv_deconv=unetConvDeconv_args,
                 double_unet=doubleunet_args
                 # other models
                 )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
