# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .encoder_decoder import EncoderDecoder

from .new_encoder_decoder import NewEncoderDecoder
__all__ = ['BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'NewEncoderDecoder']
