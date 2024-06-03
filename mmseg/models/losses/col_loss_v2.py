from ..builder import LOSSES
import torch
import torch.nn as nn


@LOSSES.register_module()
class ColorProcessingLossV2(nn.Module):
    def __init__(self,
                 loss_weight=1,
                 loss_name='loss_col_v2',
                 avg_non_ignore=False
                 ):
        super(ColorProcessingLossV2, self).__init__()
        self._loss_name = loss_name
        self.Loss_weight = loss_weight
        self.avg_non_ignore = avg_non_ignore
        self.illumination_class_cityscape = [
            86.46051320057052
            , 79.37014543897092
            , 95.30679177391578
            , 71.11888521745776
            , 75.57026559270716
            , 77.90493757655786
            , 74.77466800282637
            , 88.27701037425895
            , 57.685269557270146
            , 72.71472387765841
            , 229.9589238353863
            , 66.9194012998903
            , 60.42471796718752
            , 76.8407421534007
            , 74.98657626719087
            , 73.56771430328095
            , 123.92515568872523
            , 68.93476495876828
            , 76.0970460111028]
        self.illumination_class_nightcity = [
            76.5113984140019
            , 76.23163212875781
            , 60.90662084364415
            , 69.06930071129905
            , 69.63671393061327
            , 73.11413822794262
            , 140.7827781957324
            , 116.29554873008291
            , 46.23329954488532
            , 57.839322341112386
            , 32.61465346757989
            , 57.4385179294615
            , 62.234896087294814
            , 90.90285758569436
            , 91.99610158117673
            , 91.82209397173472
            , 94.06478985576457
            , 74.6924145472464
            , 69.15034088822232]

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

    def forward(self, reflectance, org_img, seg_label):

        illumination = torch.max(org_img, dim=1, keepdim=True)[0]
        for i in range(19):
            illumination[seg_label == i] -= self.illumination_class_nightcity[i]/255
        # 对illumination归一化到0-1
        illumination = (illumination-illumination.min()) / (illumination.max()-illumination.min())

        # illumination = torch.clamp(illumination, 0, 1)
        illumination = self.gamma(illumination, 0.4)
        for i in range(19):
            illumination[seg_label == i] += self.illumination_class_cityscape[i]/255

        illumination = torch.clamp(illumination, 0, 1)
        bright_img = reflectance * illumination
        # print("bright_img", bright_img.max().item(), bright_img.min().item())
        mean = torch.mean(bright_img, dim=[0, 2, 3], keepdim=True)
        mean = mean.squeeze()
        red_mean = mean[0]
        green_mean = mean[1]
        blue_mean = mean[2]
        loss_red_green = torch.pow(red_mean - green_mean, 2)
        loss_red_blue = torch.pow(red_mean - blue_mean, 2)
        loss_green_blue = torch.pow(green_mean - blue_mean, 2)
        loss = self.Loss_weight * (loss_red_green + loss_red_blue + loss_green_blue)

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
