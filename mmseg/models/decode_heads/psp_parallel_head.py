# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils.transforms import denorm, get_mean_std

from ..losses import accuracy

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        **kwargs)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


@HEADS.register_module()
class PSPParallelHead(BaseDecodeHead):
    """Pyramid Scene Parsing Network.

    This head is the implementation of
    `PSPNet <https://arxiv.org/abs/1612.01105>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSPParallelHead, self).__init__(**kwargs)
        assert isinstance(pool_scales, (list, tuple))
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.ref_psp_modules = PPM(
            self.pool_scales,
            self.in_channels,
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)

        self.bottleneck = ConvModule(
            self.in_channels + 2 * len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.reflectance_bottle= ConvModule(
            in_channels=len(pool_scales) * self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)

        self.reflectance_bottleneck = ConvModule(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
            norm_cfg=self.norm_cfg)
        self.conv_layer = nn.Conv2d(in_channels=self.channels, out_channels=3, kernel_size=1, stride=1, padding=0)


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        x_clone = x.clone()
        ref_out = self.ref_psp_modules(x_clone)
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs.extend(ref_out)
        psp_outs = torch.cat(psp_outs, dim=1)
        ref_out = torch.cat(ref_out, dim=1)
        feats = self.bottleneck(psp_outs)
        return feats, ref_out

    def forward(self, inputs):
        """Forward function."""
        # print("inputs", inputs[0].shape)
        # print(len(inputs))
        features = inputs[0]
        img = inputs[1]
        # print("img", img.shape)

        output, ref_out = self._forward_feature(features)
        output = self.cls_seg(output)
        ref = self.reflectance_bottle(ref_out)
        ref = self.reflectance_bottleneck(ref)
        ref = self.conv_layer(ref)

        # 将ref 上采样到和img一样的尺寸
        ref = resize(
            ref,
            size=img.size()[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return output, ref, img

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

            # seg_label = seg_label.squeeze(1)
            # org_img = org_img.squeeze(1)

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
            loss['acc_seg'] = accuracy(
                seg_logit, seg_label.squeeze(1), ignore_index=self.ignore_index)
            # loss['acc_tv'] = accuracy(
            #     reflectance_img, org_img, ignore_index=self.ignore_index)

        return loss