import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize


import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_world_size
from datasets.cuhkpedes_merge import CUHKPEDES_M
from datasets.icfgpedes import ICFGPEDES
from datasets.bases import ImageDataset, SketchDataset, ImageTextMSMDataset, ImageTextMSMMLMDataset, TextDataset, ImageTextDataset, ImageTextMCQDataset, ImageTextMaskColorDataset, ImageTextMLMDataset, ImageTextMCQMLMDataset, SketchTextDataset

root= '/data0/data_ccq/CUHK-PEDES/'
dataset = CUHKPEDES_M(root=root)

print('finished!')
