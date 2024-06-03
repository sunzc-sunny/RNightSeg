import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule

from mmseg.ops import resize

from .sep_aspp_head import DepthwiseSeparableASPPHead
from ..builder import HEADS



@HEADS.register_module()
# class EsmapDepthwiseSeparableASPPHead(DepthwiseSeparableASPPHead):
#     def __init__(self, **kwargs):
#         super(EsmapDepthwiseSeparableASPPHead, self).__init__(**kwargs)
#
#
#         self.es  = nn.Sequential(
#             DepthwiseSeparableConvModule(
#                 5 * self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg),
#             DepthwiseSeparableConvModule(
#                 self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg))
#         # 重写self.bottleneck
#
#         self.sep_bottleneck = nn.Sequential(
#             DepthwiseSeparableConvModule(
#                 self.channels + self.c1_channels + self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg),
#             DepthwiseSeparableConvModule(
#                 self.channels,
#                 self.channels,
#                 3,
#                 padding=1,
#                 norm_cfg=self.norm_cfg,
#                 act_cfg=self.act_cfg))
#
#     def forward(self, inputs):
#         """Forward function."""
#         x = self._transform_inputs(inputs)
#         aspp_outs = [
#             resize(
#                 self.image_pool(x),
#                 size=x.size()[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners)
#         ]
#         aspp_outs.extend(self.aspp_modules(x))
#         aspp_outs = torch.cat(aspp_outs, dim=1)
#         output = self.bottleneck(aspp_outs)
#         es_map = self.es(aspp_outs)
#
#         if self.c1_bottleneck is not None:
#             c1_output = self.c1_bottleneck(inputs[0])
#             output = resize(
#                 input=output,
#                 size=c1_output.shape[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners)
#             es_map = resize(
#                 input=es_map,
#                 size=c1_output.shape[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners)
#             output = torch.cat([output, c1_output, es_map], dim=1)
#         output = self.sep_bottleneck(output)
#         output = self.cls_seg(output)
#         return output, es_map
class EsmapDepthwiseSeparableASPPHead(DepthwiseSeparableASPPHead):
    def __init__(self, **kwargs):
        super(EsmapDepthwiseSeparableASPPHead, self).__init__(**kwargs)

        self.es  = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.c1_in_channels,
                self.c1_in_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.c1_in_channels,
                self.c1_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        # 重写self.bottleneck

        self.sep_bottleneck = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + self.c1_channels + self.c1_channels,
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

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)
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
        es_map = self.es(inputs[0])
        # print("inputs[0].shape", inputs[0].shape)
        # print("input.shape", inputs.shape)

        if self.c1_bottleneck is not None:
            c1_output = self.c1_bottleneck(inputs[0])
            output = resize(
                input=output,
                size=c1_output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            # es_map = resize(
            #     input=es_map,
            #     size=c1_output.shape[2:],
            #     mode='bilinear',
            #     align_corners=self.align_corners)
            output = torch.cat([output, c1_output, es_map], dim=1)
        output = self.sep_bottleneck(output)
        output = self.cls_seg(output)
        return output, es_map