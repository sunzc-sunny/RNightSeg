from copy import deepcopy

import mmcv

from mmseg.models.train_strategy.fd_decorator import TSDecorator, get_module
from mmseg.models import TS, build_segmentor
from mmseg.utils.utils import downscale_label_ratio
from mmseg.core import add_prefix
from mmseg.models.utils.visualization import subplotimg
from matplotlib import pyplot as plt

import torch
import math
import os

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0

def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


@TS.register_module()
class FD_TS(TSDecorator):
    def __init__(self, **cfg):
        super(FD_TS, self).__init__(**cfg)
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.debug_img_interval = cfg['debug_img_interval']
        self.local_iter = 0

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

    def get_imnet_model(self):
        # model = get_module(self.imnet_model)
        # 输出model 第一层参数
        # for name, param in model.named_parameters():
        #     print(name, param)

        return get_module(self.imnet_model)


    def train_step(self, data_batch, optimizer, **kwargs):

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs



    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        if mask is not None:
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            if pw_feat_dist.shape[0] == 0:
                pw_feat_dist = torch.zeros(1).to(pw_feat_dist.device).requires_grad_(True)
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        with torch.no_grad():
            self.get_imnet_model().eval()
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
        feat_loss, feat_log = self._parse_losses({'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def forward_train(self, img, img_metas, gt_semantic_seg):

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        means, stds = get_mean_std(img_metas, dev)
        #
        clean_losses = self.get_model().forward_train(img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat = clean_losses.pop('features')
        # # seg_debug['img'] = self.get_model().decode_head.debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        # # print('clean_loss', clean_loss)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg Grad: {grad_mag:.4f}', 'mmseg')

        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, src_feat)

            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Feat Grad: {grad_mag:.4f}', 'mmseg')

        # generate predication

        self.local_iter += 1
        if self.local_iter % self.debug_img_interval == 0:
            pred = self.simple_test(img, img_metas, rescale=True)
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)

            for j in range(batch_size):
                rows, cols = 2, 3
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )

                subplotimg(axs[0][0], vis_img[j], 'Input Image')
                subplotimg(axs[0][1], gt_semantic_seg[j], 'GT', cmap='cityscapes')
                subplotimg(axs[0][2], pred[j], 'Pred', cmap='cityscapes')
                if self.debug_fdist_mask is not None:
                    subplotimg(axs[1][0], self.debug_fdist_mask[j][0], 'FD Mask', cmap='gray')
                    subplotimg(axs[1][1], self.debug_gt_rescaled[j], 'GT Rescaled', cmap='cityscapes')

                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()



        return log_vars

