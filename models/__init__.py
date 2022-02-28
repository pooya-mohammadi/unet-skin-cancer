from deep_utils import log_print

from .unet_res50 import UnetRes50
from .unet_conv_deconv import UnetConvDeconv
from .unet_attention_gate import GateUnet
from .doubleunet import DoubleUnet
from .unet_cbam import CbamUnet
from .unet_cbam_gate import Cbam_Gate_Unet
from .unet_pyramid_cbam_gate import Pyramid_Cbam_Gate_Unet
from .mcgu_net import MCGUNET
from .r2unet import R2Unet
from .r2unet_cbam import R2Unet_CBAM

MODELS = dict(unet_res50=UnetRes50,
              unet_conv_deconv=UnetConvDeconv,
              unet_attention_gate=GateUnet,
              double_unet=DoubleUnet,
              unet_cbam=CbamUnet,
              unet_cbam_gate=Cbam_Gate_Unet,
              unet_pyramid_cbam_gate=Pyramid_Cbam_Gate_Unet,
              mcg_unet=MCGUNET,
              r2unet=R2Unet,
              r2unet_cbam=R2Unet_CBAM
              )


def load_model(model_name, logger=None, **kwargs):
    """Get models"""
    model = MODELS[model_name](**kwargs).get_model()
    log_print(logger, f"model: {model_name} is successfully loaded!")
    return model
