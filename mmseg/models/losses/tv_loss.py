import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register_module()
class L_TV(nn.Module):
    def __init__(self,
                 loss_weight=1,
                 loss_name='loss_tv',
                 avg_non_ignore=False):
        super(L_TV,self).__init__()

        self._loss_name = loss_name
        self.TVLoss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size


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