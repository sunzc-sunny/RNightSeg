import os

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import force_fp32

from mmseg.ops import resize

from .sep_aspp_head import DepthwiseSeparableASPPHead
from ..builder import HEADS
from ..losses import accuracy

from mmseg.models.utils.transforms import denorm, get_mean_std
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
class SegASPPModel(DepthwiseSeparableASPPHead):
    def __init__(self, **kwargs):
        super(SegASPPModel, self).__init__(**kwargs)

        self.local_iter = 0


        self.reflectance_bottle = DepthwiseSeparableConvModule(
                self.channels + self.c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        self.reflectance_bottleneck = DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)

        # 一个1x1的conv
        self.fuse_layer_1 = nn.Conv2d(in_channels= self.channels+self.channels+self.c1_channels, out_channels= self.channels+self.c1_channels, kernel_size=1, stride=1, padding=0)
        self.fuse_layer_2 = nn.Conv2d(in_channels= 2*self.channels+2*self.c1_channels, out_channels= self.channels+self.c1_channels, kernel_size=1, stride=1, padding=0)


        # 重写self.bottleneck
        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + self.c1_channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))

        self.conv_layer = nn.Conv2d(in_channels=self.channels, out_channels=3, kernel_size=1, stride=1, padding=0)

        # self.sigmoid = nn.Sigmoid()

        self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=self.channels+self.c1_channels, outchannels=self.channels+self.c1_channels)


    # def reflectance_tail(self, feat):
    #     if self.dropout is not None:
    #         feat = self.dropout(feat)
    #     output = self.conv_layer(feat)
    #     # output = self.sigmoid(output)
    #     return output

    # def gamma(self, img, gamma):
    #     i_max = img.max()
    #     # 对tensor img 做gamma 变换
    #     img = img / i_max
    #     img = torch.pow(img, gamma)
    #     img = img * i_max

    #     return img



    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training."""
        seg_logit = self.forward(inputs)
        losses = self.losses(seg_logit, gt_semantic_seg, img_metas)

        return losses

    def forward(self, inputs):
        """Forward function."""
        feature = inputs[0]
        img = inputs[1]

        x = self._transform_inputs(feature)

        aspp_outs = [
            resize(
                self.image_pool(x),
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        ]

        aspp_outs.extend(self.aspp_modules(x))
        aspp_outs = torch.cat(aspp_outs, dim=1)

        output = self.bottleneck(aspp_outs)

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(feature[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            output = torch.cat([output, c1_output], dim=1)
        # deep cpoy output
        output_clone = output.clone()
        reflectance = self.reflectance_bottle(output_clone)
        reflectance_clone = reflectance.clone()
        reflectance = self.reflectance_bottleneck(reflectance)

        # print(output.shape)
        # print(reflectance.shape)
        # print(torch.cat([output, reflectance], dim=1).shape)
        # 使用self.fuse_layer 融合output和reflectance
        # """"""""""""""""""""""""""" #
        dual_att_input = self.fuse_layer_1(torch.cat([output, reflectance_clone], dim=1))
        dual_att_output = self.DualAtt_ConBlock(dual_att_input)
        output = self.fuse_layer_2(torch.cat([dual_att_output, output], dim=1))

        output = self.sep_bottleneck(output)
        # output = self.sep_bottleneck(torch.cat([output, reflectance], dim=1))

        output = self.cls_seg(output)

        # reflectance_output = self.reflectance_tail(reflectance)

        # reflectance_output = resize(reflectance_output, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # 创建和output一样大小的tensor
        reflectance_output = torch.zeros(output.shape).cuda()
        # reflectance_output = torch.tensor()
        return output, reflectance_output, img

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




