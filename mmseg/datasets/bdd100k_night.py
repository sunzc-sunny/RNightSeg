
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100KNightDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(BDD100KNightDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
        # self.valid_mask_size = [512, 1024]
