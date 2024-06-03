# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from matplotlib import pyplot as plt

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmseg.models.utils.transforms import denorm, get_mean_std
from ..losses import accuracy

from mmseg.models.utils.visualization import subplotimg
from .uper_head_parallel_new import  UPerParallelHeadNewFuse

@HEADS.register_module()
class UPerParallelHeadTest(UPerParallelHeadNewFuse):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6),  **kwargs):
        super(UPerParallelHeadTest, self).__init__(**kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # psp module parallel
        self.psp_modules_parallel = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck_parallel = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        # FPN Module parallel
        self.lateral_convs_parallel = nn.ModuleList()
        self.fpn_convs_parallel = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs_parallel.append(l_conv)
            self.fpn_convs_parallel.append(fpn_conv)

        self.fpn_bottleneck_parallel = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv_layer = nn.Conv2d(in_channels=self.channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.local_iter = 0
    def gamma(self, img, gamma):
        i_max = img.max()
        # 对tensor img 做gamma 变换
        img = img / i_max
        img = torch.pow(img, gamma)
        img = img * i_max

        return img

    def forward_test(self, inputs, img_metas, test_cfg):
        output, reflectance_output, img = self.forward(inputs)
        illumination_class_cityscape = [
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
        illumination_class_nightcity = [
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
        d_value = torch.tensor([9.949114786568614
                                   , 3.1385133102131135
                                   , 34.40017093027163
                                   , 2.049584506158709
                                   , 5.93355166209389
                                   , 4.790799348615238
                                   , -66.00811019290602
                                   , -28.018538355823964
                                   , 11.451970012384827
                                   , 14.875401536546029, 197.3442703678064, 9.4808833704288, -1.8101781201072953,
                                -14.062115432293666, -17.00952531398586, -18.254379668453765, 29.860365832960653,
                                -5.757649588478117, 6.946705122880488])
        # print(img_metas)
        if self.local_iter % 10 == 0:
            pred = output.clone()
            pred = resize(pred, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
            topk = 1
            topk = (topk,)
            maxk = max(topk)

            pred_value, pred_label = pred.topk(maxk, dim=1)

            dev = img.device
            means, stds = get_mean_std(img_metas, dev)

            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_ref = torch.clamp(denorm(reflectance_output, means, stds), 0, 1)
            illumination = torch.max(vis_img, dim=1, keepdim=True)[0]

            for i in range(19):
                illumination[pred_label == i] = illumination[pred_label == i] / (
                        illumination_class_nightcity[i] / 255)

            new_illumination = self.gamma(illumination, 0.5)
            for i in range(19):
                new_illumination[pred_label == i] = new_illumination[pred_label == i] * (
                        illumination_class_cityscape[i] / 255)

            bright_img = vis_ref * new_illumination

            vis_bright = torch.clamp(bright_img, 0, 1)

            out_dir = './visualization/upernet_pipeline_new2'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img_name = img_metas[0]['filename'].split('/')[-1]

            row, clone = 2, 3
            batchsize = img.shape[0]

            for j in range(batchsize):
                fig, axs = plt.subplots(
                    row,
                    clone,
                    figsize=(clone * 5, row * 5),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0, 0], vis_img[j], 'source img')
                subplotimg(axs[0, 1], vis_ref[j], 'reflectance')
                subplotimg(axs[0, 2], vis_bright[j], 'bright img')

                subplotimg(axs[1, 0], pred_label[j], 'pred', cmap='cityscapes')
                subplotimg(axs[1, 1], new_illumination[j], 'new_illumination', cmap='gray')
                subplotimg(axs[1, 2], illumination[j], 'illumination', cmap='gray')

                plt.savefig(os.path.join(out_dir, img_name))
                plt.close()

        self.local_iter += 1
        return output
