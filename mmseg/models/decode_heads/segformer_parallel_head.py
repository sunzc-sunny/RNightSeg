import os
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from ..losses import accuracy
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

from mmseg.models.utils.transforms import denorm, get_mean_std
from mmseg.models.utils.visualization import subplotimg
from matplotlib import pyplot as plt


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
class SegformerParallelHead(BaseDecodeHead):
    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = nn.Sequential(
            ConvModule(
                in_channels=2 * self.channels + self.channels * num_inputs,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg),
            ConvModule(
                in_channels=self.channels,
                out_channels=self.channels,
                kernel_size=3,
                padding=1,
                norm_cfg=self.norm_cfg),
        )

        self.reflectance_bottle= ConvModule(
            in_channels=self.channels * num_inputs,
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

        # self.sigmoid = nn.Sigmoid()
        self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=self.channels+self.channels * num_inputs, outchannels=self.channels+self.channels * num_inputs)



    def forward(self, feature):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = feature[0]
        img = feature[1]

        inputs = self._transform_inputs(inputs)
        outs = []

        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        outs = torch.cat(outs, dim=1)
        output_clone = outs.clone()

        reflectance = self.reflectance_bottle(output_clone)
        reflectance_clone = reflectance.clone()
        reflectance = self.reflectance_bottleneck(reflectance)
        reflectance_output = self.conv_layer(reflectance)
        # reflectance_output = self.sigmoid(reflectance_output)

        reflectance_output = resize(reflectance_output, size=img.shape[2:], mode='bilinear', align_corners=self.align_corners)

        dual_att_input = torch.cat([reflectance_clone, output_clone], dim=1)

        dual_att_output = self.DualAtt_ConBlock(dual_att_input)

        outs = torch.cat([dual_att_output, reflectance_clone], dim=1)

        out = self.fusion_conv(outs)

        out = self.cls_seg(out)

        return out, reflectance_output, img

    def forward_test(self, inputs, img_metas, test_cfg):
        out, reflectance_output, img = self.forward(inputs)

        return out

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
