from .unet_res50 import UnetRes50


MODELS = dict(unet=UnetRes50,
              # other models
              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
