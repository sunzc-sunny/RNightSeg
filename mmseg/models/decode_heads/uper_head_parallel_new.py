# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmseg.models.utils.transforms import denorm, get_mean_std
from ..losses import accuracy

from mmseg.models.utils.visualization import subplotimg


class DualAtt_ConBlock(nn.Module):
    def __init__(self, inchannels=256, outchannels=256):
        super(DualAtt_ConBlock, self).__init__()
        self.spatialAttn = nn.Sequential(
            nn.Conv2d(outchannels, int(outchannels / 4), kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(int(outchannels / 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(outchannels / 4), out_channels=1, kernel_size=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.channelAttn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(outchannels, outchannels // 16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels // 16, outchannels, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.c3x3rb = nn.Sequential(nn.Conv2d(inchannels,
                                              outchannels,
                                              kernel_size=3,
                                              padding=1),
                                    nn.BatchNorm2d(outchannels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):  # x is a tuple of incoming feature maps at different resolutions
        fused = self.c3x3rb(x)
        spatial = self.spatialAttn(fused)
        channel = self.channelAttn(fused)
        channel = fused * channel
        fea = torch.mul(spatial.expand_as(channel) + 1, channel)  # F(X) = C(X)*(1+S(X))
        return fea

@HEADS.register_module()
class UPerParallelHeadNewFuse(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerParallelHeadNewFuse, self).__init__(
            input_transform='multiple_select', **kwargs)
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
            (len(self.in_channels)+2) * self.channels,
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

        self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=(len(self.in_channels)+1)*self.channels, outchannels=(len(self.in_channels)+1)*self.channels)

        self.conv_layer = nn.Conv2d(in_channels=self.channels, out_channels=3, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def psp_forward_parallel(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules_parallel(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck_parallel(psp_outs)

        return output

    def forward_parallel(self, inputs):

        laterals_parallel = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs_parallel)
        ]
        laterals_parallel.append(self.psp_forward_parallel(inputs))

        used_backbone_levels_parallel = len(laterals_parallel)
        for i in range(used_backbone_levels_parallel - 1, 0, -1):
            prev_shape = laterals_parallel[i - 1].shape[2:]
            laterals_parallel[i - 1] = laterals_parallel[i - 1] + resize(
                laterals_parallel[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        fpn_outs_parallel = [
            self.fpn_convs_parallel[i](laterals_parallel[i])
            for i in range(used_backbone_levels_parallel - 1)
        ]
        fpn_outs_parallel.append(laterals_parallel[-1])
        for i in range(used_backbone_levels_parallel - 1, 0, -1):
            fpn_outs_parallel[i] = resize(
                fpn_outs_parallel[i],
                size=fpn_outs_parallel[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs_parallel = torch.cat(fpn_outs_parallel, dim=1)
        feats_parallel = self.fpn_bottleneck_parallel(fpn_outs_parallel)
        return feats_parallel

    def _forward_feature(self, inputs, feats_parallel):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)

        ####### ******** ######
        fpn_outs = torch.cat([fpn_outs, feats_parallel], dim=1)
        fpn_outs = self.DualAtt_ConBlock(fpn_outs)
        fpn_outs = torch.cat([fpn_outs, feats_parallel], dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats

    def forward(self, inputs):
        """Forward function."""
        feature = inputs[0]
        img = inputs[1]

        feats_parallel = self.forward_parallel(feature)
        output = self._forward_feature(feature, feats_parallel)
        output = self.cls_seg(output)
        regression = self.conv_layer(feats_parallel)
        # regression = self.sigmoid(regression)
        reflectance_output = resize(regression, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)

        return output, reflectance_output, img

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training."""
        seg_logit = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg, img_metas)

        return losses

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logits, seg_labels, img_metas):
        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        if isinstance(seg_logits, tuple):
            seg_logit = seg_logits[0]
            reflectance_img = seg_logits[1]
            org_img = seg_logits[2]

            seg_label = seg_labels


            seg_logit = resize(
                input=seg_logit,
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

            org_img = resize(
                input=org_img,
                size=reflectance_img.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

            if self.sampler is not None:
                seg_weight = self.sampler.sample(seg_logit, seg_label)
            else:
                seg_weight = None


            dev = org_img.device
            means, stds = get_mean_std(img_metas, dev)
            # denormed image
            org_img = torch.clamp(denorm(org_img, means, stds), 0, 1)


            for loss_decode in losses_decode:
                if loss_decode.loss_name not in loss:
                    # print("not in loss")
                    if loss_decode.loss_name == 'loss_ref':
                        loss[loss_decode.loss_name] = loss_decode(
                        reflectance_img,
                        org_img,
                        img_metas)
                    elif loss_decode.loss_name == 'loss_tv':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img
                           )
                    elif loss_decode.loss_name == 'loss_ce':
                        loss[loss_decode.loss_name] = loss_decode(
                        seg_logit,
                        seg_label.squeeze(1),
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    elif loss_decode.loss_name == 'loss_ref_v2':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_logit
                        )
                    elif loss_decode.loss_name == 'loss_ref_v3':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img
                        )
                    elif loss_decode.loss_name == 'loss_col':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_col_v2':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_col_v3':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_ref_ssim':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img
                        )
                else:
                    if loss_decode.loss_name == 'loss_ref':
                        loss[loss_decode.loss_name] += loss_decode(
                        reflectance_img,
                        org_img)
                    elif loss_decode.loss_name == 'loss_tv':
                        loss[loss_decode.loss_name] += loss_decode(
                            reflectance_img
                           )
                    elif loss_decode.loss_name == 'loss_ce':
                        loss[loss_decode.loss_name] += loss_decode(
                        seg_logit,
                        seg_label.squeeze(1),
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                    elif loss_decode.loss_name == 'loss_ref_v2':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )

                    elif loss_decode.loss_name == 'loss_col':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_ref_v3':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img
                        )
                    elif loss_decode.loss_name == 'loss_col_v2':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_col_v3':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img,
                            seg_label
                        )
                    elif loss_decode.loss_name == 'loss_ref_ssim':
                        loss[loss_decode.loss_name] = loss_decode(
                            reflectance_img,
                            org_img
                        )
            loss['acc_seg'] = accuracy(
                seg_logit, seg_label.squeeze(1), ignore_index=self.ignore_index)
            # loss['acc_tv'] = accuracy(
            #     reflectance_img, org_img, ignore_index=self.ignore_index)

        return loss