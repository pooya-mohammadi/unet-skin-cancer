from .unet_res50 import unetRes50_args
from .unet_conv_deconv import unetConvDeconv_args
from .doubleـunet import doubleunet_args
from .unet_attention_gate import unetgate_args
from .unet_cbam import unetcbam_args
from .unet_cbam_gate import unetcbamgate_args
from .unet_pyramid_cbam_gate import unetcbamgate_pyramid_args

ARGUMENTS = dict(unet_res50=unetRes50_args,
                 unet_conv_deconv=unetConvDeconv_args,
                 double_unet=doubleunet_args,
                 unet_attention_gate=unetgate_args,
                 unet_cbam=unetcbam_args,
                 unet_cbam_gate=unetcbamgate_args,
                 unet_pyramid_cbam_gate=unetcbamgate_pyramid_args
                 # other models
                 )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
