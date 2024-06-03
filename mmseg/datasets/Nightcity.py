
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class NightcityDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(NightcityDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labelIds.png',
            **kwargs)
        # self.valid_mask_size = [512, 1024]
