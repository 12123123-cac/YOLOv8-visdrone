import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union, List
from einops import rearrange, reduce
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *
from .rep_block import *
from .kernel_warehouse import KWConv
from .dynamic_snake_conv import DySnakeConv
from .ops_dcnv3.modules import DCNv3, DCNv3_DyHead
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import *
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA, LocalGlobalAttention
from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
from ..backbone.rmt import RetBlock, RelPos2d
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .deconv import DEConv
from .SMPConv import SMPConv
from .camixer import CAMixer
from .orepa import *
from .RFAconv import *
from .wtconv2d import *
from .metaformer import *
from .tsdn import DTAB, LayerNorm
from .savss import SAVSS_Layer
from ..backbone.MambaOut import GatedCNNBlock_BCHW
from .efficientvim import EfficientViMBlock, EfficientViMBlock_CGLU
from .DCMPNet import LEGM
from .transMamba import TransMambaBlock
from .EVSSM import EVS
from .DarkIR import EBlock, DBlock
from ..backbone.lsnet import SKA, LSConv, Block as LSBlock
from ..backbone.overlock import RepConvBlock
from .SFSConv import SFS_Conv
from .GroupMamba.groupmamba import Block_mamba
from .MambaVision import MambaVisionBlock
from .UMFormer import GL_VSS
from .esc import ESCBlock, ConvAttn
from .UniConvNet import UniConvBlock
from .gcconv import GCConv
from .CFBlock import CFBlock
from .CSSC import CSSC, CNCM
from .HFRB import HFRB
from .EVA import EVA

from ultralytics.utils.torch_utils import make_divisible
from timm.layers import CondConv2d, trunc_normal_, use_fused_attn, to_2tuple

class DynamicConv_Single(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        
    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x

class DynamicConv(nn.Module):
    default_act = nn.SiLU()  # default activation
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, num_experts=4):
        super().__init__()
        self.conv = nn.Sequential(
            DynamicConv_Single(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, num_experts=num_experts),
            nn.BatchNorm2d(c2),
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)

class Bottleneck_DynamicConv(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DynamicConv(c2, c2, 3)

class C2f_DynamicConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DynamicConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
