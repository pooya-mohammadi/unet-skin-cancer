from .unet_res50 import UnetRes50
from .unet_conv_deconv import Unet_conv_deconv

MODELS = dict(unet_res50 = UnetRes50,
              unet_conv_deconv = Unet_conv_deconv,
              # other models
              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
