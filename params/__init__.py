from .resnet50 import resnet50_args

ARGUMENTS = dict(resnet50=resnet50_args,
                 # other args
                 )


def get_args(model_name):
    """Get Argument Parser"""
    return ARGUMENTS[model_name]()
