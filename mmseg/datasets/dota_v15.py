
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class Dotav15Dataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(Dotav15Dataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_labellds.png',
            **kwargs)
    
    CLASSES = ('plane', 'baseball-diamond', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship',
               'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
               'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad'
                )

        # self.valid_mask_size = [512, 1024]
