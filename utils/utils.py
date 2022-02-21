#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import logging
import datetime
import numpy as np

# padle
import paddle

logger = logging.getLogger(__name__)


def get_logger(log_file=None):# {{{
    """Set logger and return it.

    If the log_file is not None, log will be written into log_file.
    Else, log will be shown in the screen.

    Args:
        log_file (str): If log_file is not None, log will be written
            into the log_file.

    Return:
        ~Logger

        * **logger**: An Logger object with customed config.

    """
    # Basic config
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Add filehandler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)

    return logger# }}}

def set_seed_logger(args, cfg):# {{{
    """Experiments preparation, e.g., fix random seed, prepare checkpoint dir
    and set logger.

    Args:
        args (parser.Argument): An parser.Argument object.
        cfg (yacs.config): An yacs.config.CfgNode object.

    Return:
        ~(Logger, str):

        * **logger**: An Logger object with customed config.
        * **save_dir**: Checkpoint dir to save models.

    """
    seed = cfg['MISC']['SEED']
    # Set random seed
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Prepare save dir
    if cfg['OUTPUT']['SAVE_NAME']:
        prefix = cfg['OUTPUT']['SAVE_NAME'] + '_'
    else:
        prefix = ''
    exp_name = prefix + datetime.datetime.now().strftime('%yY_%mM_%dD_%HH')
    save_dir = os.path.join(cfg['OUTPUT']['CHECKPOINT_DIR'], exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Build logger
    log_file = os.path.join(save_dir, 'log.txt')
    logger = get_logger(log_file)

    return logger, save_dir# }}}

def dump_cfg(cfg, cfg_file):# {{{
    """Dump config of each experiment into file for backup.

    Args:
        cfg (yacs.config): An yacs.config.CfgNode object.
        cfg_file (str): Dump config to this file.

    """
    logger.info('Dump configs into {}'.format(cfg_file))
    logger.info('Using configs: ')
    logger.info(cfg)
    with open(cfg_file, 'w') as f:
        f.write(cfg.dump())# }}}

def compute_ranks(dataset, results):# {{{
    sims = np.array([results[i] for i in range(len(dataset))])
    labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
    if dataset.has_caption_indices:
        num_captions_per_image = dataset.num_captions_per_image
    else:
        num_captions_per_image = len(dataset.image_keys) * dataset.num_captions_per_image

    sims = sims.reshape([-1, num_captions_per_image])
    labels = labels.reshape([-1, num_captions_per_image])

    # Compute i2t ranks
    i2t_ranks = []
    for sim, label in zip(sims, labels):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_image
        for r, ind in enumerate(inds):
            if label[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)

    # Compute t2i ranks
    t2i_ranks = []
    if not dataset.has_caption_indices:
        sims = np.swapaxes(sims, 0, 1)
        labels = np.swapaxes(labels, 0, 1)
        for sim, label in zip(sims, labels):
            inds = np.argsort(sim)[::-1]
            rank = num_captions_per_image
            for r, ind in enumerate(inds):
                if label[ind] == 1:
                    rank = r
                    break
            t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks# }}}


def get_retrieval_results(dataset, results):
    sims = np.array([results[i] for i in range(len(dataset))])
    images = np.array([dataset.get_result(i)[0] for i in range(len(dataset))])
    captions = np.array([dataset.get_result(i)[1] for i in range(len(dataset))])
    if dataset.has_caption_indices:
        num_captions_per_image = dataset.num_captions_per_image
    else:
        num_captions_per_image = len(dataset.image_keys) * dataset.num_captions_per_image

    sims = sims.reshape([-1, num_captions_per_image])  # num_image x num_captions
    images = images.reshape([-1, num_captions_per_image]) # num_images x num_captions
    captions = captions.reshape([-1, num_captions_per_image])  # num_images x num_captions

    # Get i2t results
    i2t_results = {}
    for i, (sim, cap) in enumerate(zip(sims, captions)):
        inds = np.argsort(sim)[::-1]
        tmp_results = []
        for ind in inds:
            tmp_results.append(cap[ind])
        i2t_results[images[i][0]] = tmp_results[:10]

    # Get t2i results
    t2i_results = {}
    if not dataset.has_caption_indices:
        sims = np.swapaxes(sims, 0, 1)  # num_captions x num_images
        images = np.swapaxes(images, 0, 1)  # num_captions x num_images
        for t, (sim, image) in enumerate(zip(sims, images)):
            inds = np.argsort(sim)[::-1]
            tmp_results = []
            for ind in inds:
                tmp_results.append(image[ind])
            t2i_results[captions[0][t]] = tmp_results
            
    return i2t_results, t2i_results

