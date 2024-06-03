# Copyright (c) OpenMMLab. All rights reserved.
from .ann_head import ANNHead
from .apc_head import APCHead
from .aspp_head import ASPPHead
from .cc_head import CCHead
from .da_head import DAHead
from .dm_head import DMHead
from .dnl_head import DNLHead
from .dpt_head import DPTHead
from .ema_head import EMAHead
from .enc_head import EncHead
from .fcn_head import FCNHead
from .fpn_head import FPNHead
from .gc_head import GCHead
from .isa_head import ISAHead
from .knet_head import IterativeDecodeHead, KernelUpdateHead, KernelUpdator
from .lraspp_head import LRASPPHead
from .nl_head import NLHead
from .ocr_head import OCRHead

from .psa_head import PSAHead
from .psp_head import PSPHead
from .segformer_head import SegformerHead
from .segmenter_mask_head import SegmenterMaskTransformerHead
from .sep_aspp_head import DepthwiseSeparableASPPHead
from .sep_fcn_head import DepthwiseSeparableFCNHead
from .setr_mla_head import SETRMLAHead
from .setr_up_head import SETRUPHead
from .uper_head import UPerHead

from .esmap_sep_aspp_head import EsmapDepthwiseSeparableASPPHead
from .parallel_decoder import ParallelASPPModel
from .segformer_parallel_head import SegformerParallelHead
from .parallel_decode_test import ParallelTest
from .uper_head_parallel import UPerParallelHead
from .uper_head_parallel_test import UPerParallelHeadTest
from .uper_head_parallel_new import UPerParallelHeadNewFuse
from .parallel_decoder_big_ref import ParallelRefASPPModel
from .segformer_test_head import SegformerParallelHeadTest
from .segformer_parallel_mask_head import SegformerParallelMaskHead
from .parallel_decoder_with_mask import ParallelASPPMaskModel
from .parallel_decode_sigmoid import ParallelASPPModelSigmoid
from .parallel_decoder_new import ParallelASPPModelNew
from .segformer_parallel_head_new import SegformerParallelHeadNew
from .psp_parallel_head import PSPParallelHead

from .parallel_decode_test_new import ParallelTestNew
from .parallel_decode_test_new2 import ParallelTestNew2

from .segformer_test_new import SegformerParallelHeadTestNew
from .aspp_cascade_head import ASPPCascadeHead
from .aspp_cascade_first_head import ASPPFirstHead
from .seg_aspp_model import SegASPPModel
__all__ = [
    'FCNHead', 'PSPHead', 'ASPPHead', 'PSAHead', 'NLHead', 'GCHead', 'CCHead',
    'UPerHead', 'DepthwiseSeparableASPPHead', 'ANNHead', 'DAHead', 'OCRHead',
    'EncHead', 'DepthwiseSeparableFCNHead', 'FPNHead', 'EMAHead', 'DNLHead',
     'APCHead', 'DMHead', 'LRASPPHead', 'SETRUPHead',
     'DPTHead', 'SETRMLAHead', 'SegmenterMaskTransformerHead',
    'SegformerHead', 'ISAHead', 'IterativeDecodeHead',
    'KernelUpdateHead', 'KernelUpdator',
    'EsmapDepthwiseSeparableASPPHead',
    'ParallelASPPModel',
    'SegformerParallelHead',
    'ParallelTest',
    'UPerParallelHead',
    'UPerParallelHeadTest',
    'UPerParallelHeadNewFuse',
    'ParallelRefASPPModel',
    'SegformerParallelHeadTest',
    'SegformerParallelMaskHead',
    'ParallelASPPMaskModel',
    'ParallelASPPModelSigmoid',
    'ParallelASPPModelNew',
    'SegformerParallelHeadNew',
    'PSPParallelHead',
    'ParallelTestNew',
    'ParallelTestNew2',
    'SegformerParallelHeadTestNew',
    'ASPPCascadeHead',
    'ASPPFirstHead',
    'SegASPPModel'

]


