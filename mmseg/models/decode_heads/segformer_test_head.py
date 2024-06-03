import numpy as np
from matplotlib import pyplot as plt

from mmseg.models.utils.visualization import *
from .segformer_parallel_head import SegformerParallelHead
from .segformer_parallel_head_new import  SegformerParallelHeadNew

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
class SegformerParallelHeadTest(SegformerParallelHeadNew):
    def __init__(self, **kwargs):
        super(SegformerParallelHeadTest, self).__init__(**kwargs)
        self.local_iter = 0

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
                in_channels=self.channels * num_inputs,
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

        self.reflectance_bottle = ConvModule(
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
        # self.fuse_layer_1 = nn.Conv2d(in_channels=self.channels + self.channels * num_inputs, out_channels=self.channels * num_inputs, kernel_size=1, stride=1, padding=0)
        # self.fuse_layer_2 = nn.Conv2d(in_channels=self.channels * num_inputs+ self.channels, out_channels=self.channels * num_inputs, kernel_size=1, stride=1, padding=0)
        # # self.sigmoid = nn.Sigmoid()
        # self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=self.channels * num_inputs,
        #                                          outchannels=self.channels * num_inputs)

        self.fuse_layer_1 = nn.Conv2d(in_channels=self.channels + self.channels * num_inputs,
                                      out_channels=self.channels * num_inputs, kernel_size=1, stride=1, padding=0)
        self.fuse_layer_2 = nn.Conv2d(in_channels=2 * self.channels * num_inputs,
                                      out_channels=self.channels * num_inputs,
                                      kernel_size=1, stride=1, padding=0)

        # self.sigmoid = nn.Sigmoid()
        self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=self.channels * num_inputs,
                                                 outchannels=self.channels * num_inputs)


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

        # print(img_metas)
        if 0 == 0:
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
            ref_org = vis_img / (illumination+0.01)
            for i in range(19):
                illumination[pred_label == i] = illumination[pred_label == i] / (
                        illumination_class_nightcity[i] / 255)

            new_illumination = self.gamma(illumination, 0.4)
            for i in range(19):
                new_illumination[pred_label == i] = new_illumination[pred_label == i] * (
                        illumination_class_cityscape[i] / 255)
            # illumination
            bright_img = vis_ref * torch.max(vis_img, dim=1, keepdim=True)[0]

            vis_bright = torch.clamp(bright_img, 0, 1)
            # result_img = vis_bright.clone()

            guide = vis_img.clone().cpu().numpy()
            src = vis_bright.clone().cpu().numpy()
            # 将guide从[1,3,512,1024]转换为[512,1024,3]
            guide = np.transpose(guide, (0, 2, 3, 1))[0]
            # 将src从[1,3,512,1024]转换为[512,1024,3]
            src = np.transpose(src, (0, 2, 3, 1))[0]

            # print(guide.shape, src.shape)
            result_img = cv2.ximgproc.guidedFilter(guide=guide, src=src, radius=30, eps=0.01, dDepth=-1)
            result_img = torch.from_numpy(result_img).permute(2, 0, 1).unsqueeze(0).to(dev)
            out_dir = './visualization/segformer_new_2'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            img_name = img_metas[0]['filename'].split('/')[-1]
            result = vis_bright.clone()
            # 使用plt保存图片vis_bright,名称为img_name
            # plt.imsave(os.path.join(out_dir, img_name), result[0].cpu().numpy().transpose(1, 2, 0))
            result0 = result[:,0,:,:]
            result1 = result[:,1,:,:]
            result2 = result[:,2,:,:]
            result_img0 = result_img[:,0,:,:]
            result_img1 = result_img[:,1,:,:]
            result_img2 = result_img[:,2,:,:]
            pos = pred_label == 10
            # 将pos转成和result_img一样的形状
            pos = pos.expand_as(result_img)
            result[pos] = result_img[pos]
            # result0[pred_label == 10] = result_img0[pred_label == 10]
            # result1[pred_label == 10] = result_img1[pred_label == 10]
            # result2[pred_label == 10] = result_img2[pred_label == 10]
            row, clone = 1, 1
            batchsize = img.shape[0]

            for j in range(batchsize):

                with torch.no_grad():
                    # img_show = new_illumination[j].cpu().numpy()
                    # img_show = img_show.squeeze(0)
                    # print(img_show.shape)
                    # img_show = vis_bright[j].cpu()
                    img_show = result_img[j].cpu()

                    img_show = img_show.permute(1, 2, 0).numpy()

                    # if torch.is_tensor(img_show):
                    #     img_show = img_show.cpu()
                    # if len(img_show.shape) == 2:
                    #     if torch.is_tensor(img_show):
                    #         img_show = img_show.numpy()
                    # elif img_show.shape[0] == 1:
                    #     if torch.is_tensor(img_show):
                    #         img_show = img_show.numpy()
                    #     img_show = img_show.squeeze(0)
                    # elif img_show.shape[0] == 3:
                    #     img_show = img_show.permute(1, 2, 0)
                    #     if not torch.is_tensor(img_show):
                    #         img_show = img_show.numpy()
                    # if kwargs.get('cmap') == 'cityscapes':
                    #     kwargs.pop('cmap')
                    # if torch.is_tensor(img_show):
                    #     img_show = img_show.numpy()
                    # img_show = colorize_mask(img_show, Cityscapes_palette)
                # fig, axs = plt.subplots(
                #     row,
                #     clone,
                #     figsize=(clone * 5, row * 5),
                #     gridspec_kw={
                #         'hspace': 0.1,
                #         'wspace': 0,
                #         'top': 0.95,
                #         'bottom': 0,
                #         'right': 1,
                #         'left': 0
                #     },
                # )

                # subplotimg(axs[0, 0], pred_label[j], 'source img')
                # subplotimg(axs[0, 1], vis_ref[j], 'reflectance')
                # subplotimg(axs[0, 2], vis_bright[j], 'bright img')
                #
                # subplotimg(axs[1, 0], pred_label[j], 'pred', cmap='cityscapes')
                # subplotimg(axs[1, 1], new_illumination[j], 'new_illumination', cmap='gray')
                # subplotimg(axs[1, 2], illumination[j], 'illumination', cmap='gray')
                #
                # subplotimg(axs[2, 0], ref_org[j], 'ref gt')
                # subplotimg(axs[2, 1], result_img[j], 'result img')
                # subplotimg(axs[2, 2], result[j], 'result')
                # print(os.path.join(out_dir, img_name))
                    ...  # 图片代码
                    plt.axis('off')  # 去坐标轴
                    plt.xticks([])  # 去刻度
                    plt.imshow(img_show)
                    # plt.savefig('xxx.jpg', bbox_inches='tight', pad_inches=-0.1)  # 注意两个参数
                    plt.show()
                    img_name = "enhance_" + img_name
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
