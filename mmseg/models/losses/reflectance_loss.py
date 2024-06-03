from ..builder import LOSSES
import torch
import torch.nn as nn
from mmseg.models.utils.transforms import denorm, get_mean_std



@LOSSES.register_module()
class ReflectanceLoss(nn.Module):
    def __init__(self,
                 loss_weight=1,
                 loss_name='loss_ref',
                 avg_non_ignore=False
                 ):
        super(ReflectanceLoss, self).__init__()
        self._loss_name = loss_name
        self.Loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def gamma(self, img, gamma):
        i_max = img.max()
        # 对tensor img 做gamma 变换
        img = img / i_max
        img = torch.pow(img, gamma)
        img = img * i_max

        return img

    def forward(self, reflectance, org_img, img_metas):
        # illumination = torch.max(img, dim=1, keepdim=True)[0]

        illumination = torch.max(org_img, dim=1, keepdim=True)[0]
        # pred = torch.clamp(denorm(pred, means, stds), 0, 1)
        new_illumination = self.gamma(illumination, 0.4)
        # pred与 illumination 逐像素点相乘
        img_pred = reflectance * new_illumination
        # L2 loss
        loss_l2 = torch.mean(torch.pow(org_img - img_pred, 2))
        loss = self.Loss_weight * loss_l2
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name
