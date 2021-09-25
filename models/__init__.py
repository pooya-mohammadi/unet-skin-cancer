from .unet_res50 import UnetRes50
from .unet_conv_deconv import UnetConvDeconv

MODELS = dict(unet_res50 = UnetRes50,
              unet_conv_deconv = UnetConvDeconv,
              # other models
              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
