from .unet_res50 import UnetRes50
from .unet_conv_deconv import UnetConvDeconv
from .unet_attention_gate import GateUnet
from .doubleunet import DoubleUnet
from .unet_cbam import CbamUnet
from .unet_cbam_gate import Cbam_Gate_Unet
from .unet_pyramid_cbam_gate import Pyramid_Cbam_Gate_Unet
from .mcgu_net import MCGUNET

MODELS = dict(unet_res50=UnetRes50,
              unet_conv_deconv=UnetConvDeconv,
              unet_attention_gate=GateUnet,
              double_unet=DoubleUnet,
              unet_cbam=CbamUnet,
              unet_cbam_gate=Cbam_Gate_Unet,
              unet_pyramid_cbam_gate=Pyramid_Cbam_Gate_Unet,
              mcg_unet=MCGUNET
              # other models
              )


def load_model(model_name, **kwargs):
    """Get models"""
    return MODELS[model_name](**kwargs).get_model()
