from .fuser import ConcatFusion, CrossAttentionFusion
from mmcv.utils import Registry

FUSERS = Registry('fuser')
FUSERS.register_module()(ConcatFusion)
FUSERS.register_module()(CrossAttentionFusion)