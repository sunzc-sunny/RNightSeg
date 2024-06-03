from ..builder import LOSSES
import torch
import torch.nn as nn


@LOSSES.register_module()
class ReflectanceIllLoss():