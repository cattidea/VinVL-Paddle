#! /usr/bin/env python
# -*- coding: utf-8 -*-

import paddle
from paddle.optimizer import AdamW

def set_optimizer(model, lr_scheduler, cfg):
    epsilon = cfg['OPTIMIZATION']['EPSILON']
    optimizer = cfg['OPTIMIZATION']['OPTIMIZER']
    weight_decay = cfg['OPTIMIZATION']['WEIGHT_DECAY']
    clip_max_norm = cfg['OPTIMIZATION']['CLIP_MAX_NORM']

    no_decay = ['bias', 'LayerNorm.weight']

    if optimizer == 'adamw':
        opt = AdamW(
            learning_rate=lr_scheduler,
            parameters=[p for n, p in model.named_parameters()],
            weight_decay=weight_decay,
            epsilon=epsilon,
            apply_decay_param_fun=lambda x:x in [
                p.name for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            grad_clip=paddle.fluid.clip.ClipGradByValue(clip_max_norm)
        )
    else:
        raise ValueError('Support optimizers are: [adamw], \
                but the given one is {}'.format(optimizer))
    return opt