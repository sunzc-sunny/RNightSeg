import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
from .encoder_decoder import EncoderDecoder

@SEGMENTORS.register_module()
class NewEncoderDecoder(EncoderDecoder):
    def __init__(self, *args, **kwargs):
        super(NewEncoderDecoder, self).__init__(*args, **kwargs)

    # def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg, img):
    #
    #     losses = dict()
    #     loss_decode = self.decode_head.forward_train(x, img_metas, gt_semantic_seg, img, self.train_cfg)
    #     losses.update(add_prefix(loss_decode, 'decode'))
    #     return losses
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)

        x_new = (x, img)

        out = self._decode_head_forward_test(x_new, img_metas)
        # change
        
        if isinstance(out, tuple):
            out = out[0]

        else:
            out = out
        

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out



    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)


    def forward_train(self, img, img_metas, gt_semantic_seg, return_feat=False):
        x = self.extract_feat(img)
        losses = dict()
        if return_feat:
            losses['features'] = x

        x_new = (x, img)

        loss_decode = self._decode_head_forward_train(x_new, img_metas, gt_semantic_seg)

        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses