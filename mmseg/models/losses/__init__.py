# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszLoss
from .tversky_loss import TverskyLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss

from .tv_loss import L_TV
from .reflectance_loss import ReflectanceLoss
from .reflectance_loss_illumination_update import ReflectanceIlluminationUpdateLoss
from .col_loss import ColorProcessingLoss
from .reflectance_v3 import ReflectanceLossV3
from .col_loss_v2 import ColorProcessingLossV2
from .col_loss_v3 import ColorProcessingLossV3
from .reflectance_ssim_loss import ReflectanceSSIMLoss
__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'TverskyLoss',
    'L_TV',
    'ReflectanceLoss',
    'ReflectanceIlluminationUpdateLoss',
    'ColorProcessingLoss',
    'ReflectanceLossV3',
    'ColorProcessingLossV2',
    'ColorProcessingLossV3',
    'ReflectanceSSIMLoss'
]
