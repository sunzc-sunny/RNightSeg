from copy import deepcopy

from mmcv.parallel import MMDistributedDataParallel

from mmseg.models import BaseSegmentor, build_segmentor

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


class TSDecorator(BaseSegmentor):

    def __init__(self, **cfg):
        super(BaseSegmentor, self).__init__()

        self.model = build_segmentor(deepcopy(cfg['model']))
        self.train_cfg = cfg['model']['train_cfg']
        self.test_cfg = cfg['model']['test_cfg']
        self.num_classes = cfg['model']['decode_head']['num_classes']

    def get_model(self):
        return get_module(self.model)

    def extract_feat(self, img):
        """Extract features from images."""
        return self.get_model().extract_feat(img)

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_model().encode_decode(img, img_metas)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      return_feat=False,
                      ):

        losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=return_feat)
        return losses

    def inference(self, img, img_metas, rescale):
        """Inference image(s) with the segmentor.
        Args:
            img (torch.Tensor | np.ndarray): The input image.
            img_metas (list[dict]): List of image information.
            rescale (bool): Whether rescale back to original shape.
        Returns:
            list[np.ndarray]: The segmentation result.
        """
        return self.get_model().inference(img, img_metas, rescale)

    def simple_test(self, img, img_meta, rescale=True):
        """Test function without test time augmentation.
        Args:
            img (torch.Tensor): Input images.
            img_meta (list[dict]): List of image information.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
        Returns:
            list[np.ndarray]: Segmentation results of the images.
        """
        return self.get_model().simple_test(img, img_meta, rescale)

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.
        Args:
            imgs (torch.Tensor): Input images.
            img_metas (list[dict]): List of image information.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
        Returns:
            list[np.ndarray]: Segmentation results of the images.
        """
        return self.get_model().aug_test(imgs, img_metas, rescale)