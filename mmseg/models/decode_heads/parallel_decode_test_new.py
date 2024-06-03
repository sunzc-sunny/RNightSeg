import os

import cv2
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
from .parallel_decoder import ParallelASPPModel
from .parallel_decode_sigmoid import ParallelASPPModelSigmoid


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
class ParallelTestNew(ParallelASPPModel):
    def __init__(self, **kwargs):
        super(ParallelASPPModel, self).__init__(**kwargs)

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
        self.fuse_layer_1 = nn.Conv2d(in_channels=self.channels + self.channels + self.c1_channels,
                                      out_channels=self.channels + self.c1_channels, kernel_size=1, stride=1, padding=0)
        self.fuse_layer_2 = nn.Conv2d(in_channels=2 * self.channels + 2 * self.c1_channels,
                                      out_channels=self.channels + self.c1_channels, kernel_size=1, stride=1, padding=0)

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

        self.DualAtt_ConBlock = DualAtt_ConBlock(inchannels=self.channels + self.c1_channels,
                                                 outchannels=self.channels + self.c1_channels)


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
                , 67.685269557270146
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

                new_illumination = self.gamma(illumination, 0.5)
                for i in range(19):
                    new_illumination[pred_label == i] = new_illumination[pred_label == i] * (
                            illumination_class_cityscape[i] / 255)

                bright_img = vis_ref * new_illumination

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
                out_dir = './visualization/deeplabv3_ref_train'
                print(out_dir)
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
                # row, clone = 3, 3
                row, clone = 1, 1
                batchsize = img.shape[0]

                for j in range(batchsize):

                    with torch.no_grad():
                        image_show = vis_ref[j].cpu().numpy().transpose(1, 2, 0)
                        plt.axis('off')
                        plt.xticks([])

                        plt.imshow(image_show)
                        plt.savefig(os.path.join(out_dir, img_name), bbox_inches='tight', pad_inches=0)
                        plt.close()
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
                    # subplotimg(axs[0, 0], vis_img[j], 'source img')
                    # subplotimg(axs[0, 1], vis_ref[j], 'reflectance')
                    # subplotimg(axs[0, 2], vis_bright[j], 'bright img')

                    # subplotimg(axs[1, 0], pred_label[j], 'pred', cmap='cityscapes')
                    # subplotimg(axs[1, 1], new_illumination[j], 'new_illumination', cmap='gray')
                    # subplotimg(axs[1, 2], illumination[j], 'illumination', cmap='gray')

                    # subplotimg(axs[2, 0], ref_org[j], 'ref gt')
                    # subplotimg(axs[2, 1], result_img[j], 'result img')
                    # subplotimg(axs[2, 2], result[j], 'result')
                    # # print(os.path.join(out_dir, img_name))


                    # # plt.imshow()
                    # # # 设置plt保存图像的坐标轴不可见
                    # # plt.axis('off')
                    # # # plt.savefig(os.path.join(out_dir, img_name), bbox_inches='tight', pad_inches=0)
                    # #
                    # #
                    # #
                    # #
                    # # plt.axis('off')
                    # # # plt.xticks([])
                    # # # plt.yticks([])
                    # # plt.tight_layout()
                    # plt.savefig(os.path.join(out_dir, img_name), dpi=600)

            return output