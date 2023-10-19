import torch
from functools import partial
from easydict import EasyDict as edict
from albumentations import *

from ...data.datasets import *
from ...model.losses import *
from ...data.transforms import *
from ...engine.trainer import ISTrainer
from ...model.metrics import AdaptiveIoU
from ...data.points_sampler import MultiPointSampler
from ...utils.log import logger
from ...model import initializer

from ...model.is_hrnet_model import HRNetModel
from ...model.is_deeplab_model import DeeplabModel