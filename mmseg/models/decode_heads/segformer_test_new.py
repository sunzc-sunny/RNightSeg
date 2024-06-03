import numpy as np
from matplotlib import pyplot as plt

from mmseg.models.utils.visualization import *
from .segformer_parallel_head import SegformerParallelHead
from .segformer_parallel_head_new import  SegformerParallelHeadNew
from .segformer_head import SegformerHead
import os
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32
from ..losses import accuracy
from ..builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize

from mmseg.models.utils.transforms import denorm, get_mean_std
import cv2

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize




@HEADS.register_module()
class SegformerParallelHeadTestNew(BaseDecodeHead):
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

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def gamma(self, img, gamma):
        i_max = img.max()
        # 对tensor img 做gamma 变换
        img = img / i_max
        img = torch.pow(img, gamma)
        img = img * i_max

        return img

    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
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

        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        return out

    def forward_test(self, inputs, img_metas, test_cfg):
        print(len(inputs))
        output = self.forward(inputs)

        # print(img_metas)
        if 0 == 0:
            pred = output.clone()
            pred = resize(pred, size=output.shape[2:], mode='bilinear', align_corners=self.align_corners)
            topk = 1
            topk = (topk,)
            maxk = max(topk)

            pred_value, pred_label = pred.topk(maxk, dim=1)

            dev = output.device
            out_dir = './visualization/segformer_org_show'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img_name = img_metas[0]['filename'].split('/')[-1]

            batchsize = output.shape[0]

            for j in range(batchsize):

                with torch.no_grad():
                    img_show = pred_label[j].cpu().numpy()
                    img_show = img_show.squeeze(0)
                    print(img_show.shape)

                    img_show = colorize_mask(img_show, Cityscapes_palette)
                    plt.figure(figsize=(1024, 512))
                    plt.axis('off')  # 去坐标轴
                    plt.xticks([])  # 去刻度
                    plt.imshow(img_show)
                    # plt.savefig('xxx.jpg', bbox_inches='tight', pad_inches=-0.1)  # 注意两个参数
                    plt.show()
                    img_name = "result_" + img_name
                    plt.savefig(os.path.join(out_dir, img_name), bbox_inches='tight', pad_inches=-0.1, dpi=300)  # 注意两个参数

                    # plt.savefig()
                    plt.close()

        return output
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
