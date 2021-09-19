from .resnet50 import UNET1


MODELS = dict(unet=UNET1,
              # other models
              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
