from .unet import UnetRes50
ARGUMENTS = dict(unet = UnetRes50
                 # other args
                 )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
