# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class RoadAnomalyDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('background','anomaly')

    PALETTE = [[0, 0, 0], [255, 0, 0]]

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.labels/labels_semantic.png',
                 **kwargs):
        
        super(RoadAnomalyDataset, self).__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=False, **kwargs)
        self.custom_classes = True
        self.label_map = {0: 0, 2: 1}
