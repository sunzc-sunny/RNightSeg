# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn
import torch
from copy import deepcopy
import mmcv
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.parallel import MMDistributedDataParallel
from mmseg.utils.utils import downscale_label_ratio
from mmseg.models import build_segmentor, BaseSegmentor
# from mmseg.models.utils.visualization import subplotimg


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


# @SEGMENTORS.register_module()
class EncoderDecoder_fd(BaseSegmentor):

    def __init__(self, **cfg):
        super(EncoderDecoder_fd, self).__init__(**cfg)
        print(cfg['model']['imnet_feature_dist_classes'])

        self.model = build_segmentor(deepcopy(cfg['model']))

        self.fdist_classes = cfg['model']['imnet_feature_dist_classes']
        self.fdist_lambda = cfg['model']['imnet_feature_dist_lambda']
        self.fdist_scale_min_ratio = cfg['model']['imnet_feature_dist_scale_min_ratio']

        self.enable_fdist = self.fdist_lambda > 0
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        print(self.fdist_classes)
        print(self.fdist_lambda)
        print(self.fdist_scale_min_ratio)

# @SEGMENTORS.register_module()
# class EncoderDecoder_fd(EncoderDecoder):
#
#     def __init__(self, fdist_classes=None, fdist_lambda=0.0, fdist_scale_min_ratio=0.0, **cfg):
#         super(EncoderDecoder_fd, self).__init__(**cfg)
#
#
#         self.fdist_classes = fdist_classes
#         self.fdist_lambda = fdist_lambda
#         self.fdist_scale_min_ratio = fdist_scale_min_ratio
#
#         self.model = build_segmentor(deepcopy(cfg['model']))
#
#
#         self.enable_fdist = self.fdist_lambda > 0
#         if self.enable_fdist:
#             self.imnet_model = build_segmentor(deepcopy(cfg['model']))
#         else:
#             self.imnet_model = None
#
#         print(self.fdist_classes)
#         print(self.fdist_lambda)
#         print(self.fdist_scale_min_ratio)



    def get_model(self):
        return get_module(self.model)


    def get_imnet_model(self):
        return get_module(self.imnet_model)


    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 -f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]

            if pw_feat_dist.shape[0] == 0:
                pw_feat_dist = torch.tensor([0.0]).to(pw_feat_dist.device).requires_grad_(True)
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
            # mmcv.print_log(f'fd masked: {pw_feat_dist.mean()}', 'mmseg')

        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        with torch.no_grad():
            self.get_imnet_model().eval()  # 现在还没有imnet_model
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1

        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, dim=-1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay], fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescaled = gt_rescaled

        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_loss({'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def train_step(self, data_batch, optimizer, **kwargs):
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))  # 这个地方可能有bug 可能没有img_metas
        return outputs



    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image information.
            gt_semantic_seg (Tensor): Ground truth for semantic segmentation.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        log_vars = {}
        batch_size = img.shape(0)
        dev = img.device
        seg_debug = {}

        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        seg_debug['img'] = self.get_model().decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_loss(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enabled_fdist)

        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()

        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        # if self.debug_fdist_mask is not None:
        #     subplotimg(
        #         axs[0][0],
        #         self.debug_fdist_mask[j][0],
        #         'FDist Mask',
        #         cmap = 'gray'
        #     )
        #
        #
        #
        #     seg_debug['fd_mask'] = self.debug_fdist_mask
        #     seg_debug['gt_rescaled'] = self.debug_gt_rescaled
        # x = self.extract_feat(img)
        # losses = dict()
        #
        # # semantic segmentation loss
        # seg_logit = self.decode_head(x)
        # seg_loss, seg_log = self.decode_head.loss(seg_logit, gt_semantic_seg)
        # losses.update(seg_log)
        # losses.update({'loss_seg': seg_loss})
        #
        # # feature dist loss
        # feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, x)
        # losses.update(feat_log)
        # losses.update({'loss_feat': feat_loss})
        #
        # return losses
