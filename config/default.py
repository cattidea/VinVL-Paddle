#! /usr/bin/env python
# -*- coding: utf-8 -*-

from yacs.config import CfgNode as CN

# Create a Node
__C = CN()

# ========================== INPUT =========================
__C.INPUT = CN()
__C.INPUT.BERT_MODEL = 'bert-base-uncased'
__C.INPUT.MAX_REGION = 70
__C.INPUT.MAX_SEQ_LEN = 70
__C.INPUT.IMG_FEATURE_DIM = 2054
__C.INPUT.IMG_FEATURE_TYPE = 'frcnn'
# Whether add object detection labels as input
__C.INPUT.ADD_OD_LABEL = True
__C.INPUT.DO_LOWER_CASE = True
__C.INPUT.ATT_MASK_TYPE = 'CLR'
# Sample this number of captions for each image
__C.INPUT.NUM_CAPTIONS_PER_IMAGE_TRN = 5
__C.INPUT.NUM_CAPTIONS_PER_IMAGE_DEV = 5

# ========================== DATASET =========================
__C.DATASET = CN()
__C.DATASET.NAME = 'COCO'
__C.DATASET.DATA_DIR = ''
__C.DATASET.FEAT_FILE = ''
__C.DATASET.TRAIN = 'train'
__C.DATASET.DEV = 'minival'
__C.DATASET.TEST = 'test'

# ========================== OUPUT =========================
__C.OUTPUT = CN()
__C.OUTPUT.SAVE_NAME = ''
# Save checkpoint frequency (epochs)
__C.OUTPUT.SAVE_FREQ = 1
__C.OUTPUT.NUM_LABELS = 2
__C.OUTPUT.CHECKPOINT_DIR = './exp'

# ========================== OPTIMIZATION =========================
__C.OPTIMIZATION = CN()
__C.OPTIMIZATION.LR = 2e-5
__C.OPTIMIZATION.EPSILON = 1e-8
__C.OPTIMIZATION.LOSS_TYPE = 'sfmx'
__C.OPTIMIZATION.BATCH_SIZE = 32
__C.OPTIMIZATION.WARMUP_STEPS = 0
__C.OPTIMIZATION.LR_SCHEDULER = 'linear'
__C.OPTIMIZATION.WEIGHT_DECAY = 0.05
__C.OPTIMIZATION.EPOCHS = 30
# Clip gradients at this value
__C.OPTIMIZATION.CLIP_MAX_NORM = 1.0
__C.OPTIMIZATION.OPTIMIZER = 'adamw'
# Gradient accumulation steps
__C.OPTIMIZATION.GRADIENT_ACCUMULATION_STEPS = 1

# ========================== MONITOR =========================
__C.MONITOR = CN()
# Print training log frequency (steps)
__C.MONITOR.PRINT_STEP = 100
# Evaluation frequency (epochs)
__C.MONITOR.EVAL_FREQ = 1

# ========================== PRETRAINED =========================
__C.PRETRAINED = CN()
__C.PRETRAINED.DIR = ''
__C.PRETRAINED.RESUME = ''

# ========================== EVAL =========================
__C.EVAL = CN()
__C.EVAL.CHECKPOINT_DIR = ''
__C.EVAL.EVAL_CROSS_IMAGE = False
__C.EVAL.EVAL_IMG_KEYS_FILE = ''
__C.EVAL.EVAL_CAPTION_INDEX_FILE = ''

# ========================== MISC =========================
__C.MISC = CN()
__C.MISC.SEED = 123
__C.MISC.NUM_WORKERS = 8


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    return __C.clone()
