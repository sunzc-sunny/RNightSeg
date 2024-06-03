from ..builder import LOSSES
import torch
import torch.nn as nn
from .reflectance_loss import ReflectanceLoss
from mmseg.ops import resize


@LOSSES.register_module()
class ReflectanceIlluminationUpdateLoss(ReflectanceLoss):
    def __init__(self,
                 loss_weight=1,
                 loss_name='loss_ref_v2',
                 avg_non_ignore=False
                 ):
        super(ReflectanceIlluminationUpdateLoss, self).__init__(loss_weight, loss_name, avg_non_ignore)


    def forward(self, reflectance, img, seg_label):
        illumination = torch.max(img, dim=1, keepdim=True)[0]
        # print("illumination", illumination.max().item(), illumination.min().item())
        # print("reflectance", reflectance.max().item(), reflectance.min().item())
        # pred与 illumination 逐像素点相乘
        img_pred = reflectance * illumination
        # print("img_pred", img_pred.max().item(), img_pred.min().item())
        # L2 loss
        loss_l2 = torch.mean(torch.pow(img - img_pred, 2))
        loss = self.Loss_weight * loss_l2
        return loss
