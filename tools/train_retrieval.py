#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import random
import logging
import argparse
import numpy as np

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler

# model
from models.bert import BertConfig
from models.oscar import OscarForVLTaks
# dataset
from datasets.retrieval_dataset import RetrievalDataset
# solver
from solver.optimizer import set_optimizer
from solver.scheduler import set_scheduler
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import (
    dump_cfg,
    compute_ranks,
    set_seed_logger,
)

# logging basic config
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)


def train_epoch(model, trn_dataloader, optimizer, scheduler, epoch, logger, args, cfg):# {{{
    # Set mode for training
    model.train()
    # Set epoch for trn_sampler
    trn_dataloader.batch_sampler.set_epoch(epoch)

    logger.info('=====> Start epoch {}:'.format(epoch + 1))
    time.sleep(0.5)

    print_steps = cfg['MONITOR']['PRINT_STEP']
    grad_accum_steps = cfg['OPTIMIZATION']['GRADIENT_ACCUMULATION_STEPS']

    train_loss = 0.
    for step, (_, batch) in enumerate(trn_dataloader):
        inputs = {
            'input_ids':      paddle.concat((batch[0], batch[5]), axis=0),
            'attention_mask': paddle.concat((batch[1], batch[6]), axis=0),
            'token_type_ids': paddle.concat((batch[2], batch[7]), axis=0),
            'img_feats':      paddle.concat((batch[3], batch[8]), axis=0),
            'labels':         paddle.concat((batch[4], batch[9]), axis=0)
        }

        # Forward
        outputs = model(**inputs)
        loss, _ = outputs[:2]
        if args.n_gpus > 1:
            loss = loss.mean()
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps

        # Backward
        loss.backward()
        train_loss += float(loss)

        # Update parameters
        if (step + 1) % grad_accum_steps == 0:
            scheduler.step()
            optimizer.step()
            optimizer.clear_grad()

        # Training log
        if (step + 1) % print_steps == 0:
            logger.info('Epoch: [{}], step: [{}], lr: {:.6f}, batch_loss: {:.4f}, avg_loss: {:.4f}'.format(
                epoch + 1, step + 1, optimizer.get_lr(), float(loss), train_loss / (step + 1)
            ))

    train_loss = train_loss / (step + 1)
    logger.info('** ** Epoch [%d] done! Training loss: %.5f ** **'
                % (epoch + 1, train_loss))# }}}


def validate(model, dev_dataloader, epoch, logger):# {{{
    # Set mode for evaluation
    model.eval()

    results = {}
    with paddle.no_grad():
        for inds, batch in dev_dataloader:
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'img_feats':      batch[3],
                'labels':         batch[4],
            }
            _, logits = model(**inputs)[:2]
            probs = F.softmax(logits, axis=1)
            # The confidence to be a matched pair
            result = probs[:, 1]
            inds = [inds[i].item() for i in range(inds.shape[0])]
            result = [result[i].item() for i in range(result.shape[0])]
            results.update({ind: res for ind, res in zip(inds, result)})

    # Compute accuracy
    i2t_ranks, t2i_ranks = compute_ranks(dev_dataloader.dataset, results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]

    logger.info('** * ** Eval at Epoch [%d]! Eval Reults:  ** * **' % (epoch + 1))
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
        i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    if len(t2i_ranks) != 0:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
            t2i_accs[0], t2i_accs[1], t2i_accs[2]))# }}}


def main(args, cfg):
    # 1. Preparation
    logger, save_dir = set_seed_logger(args, cfg)
    # backup config
    cfg_file = os.path.join(save_dir, 'config.yaml')
    dump_cfg(cfg, cfg_file)
    # Fix bugs in paddle-nlp when multi-gpu training
    time.sleep(random.randint(0, 5))
    from paddlenlp.transformers.bert.tokenizer import BertTokenizer

    # 2. Create train/dev dataloader
    # dataset
    tokenizer = BertTokenizer.from_pretrained(
        cfg['INPUT']['BERT_MODEL'])
    trn_dataset = RetrievalDataset(split=cfg['DATASET']['TRAIN'],
                                   cfg=cfg,
                                   tokenizer=tokenizer,
                                   training=True)
    dev_dataset = RetrievalDataset(split=cfg['DATASET']['DEV'],
                                   cfg=cfg,
                                   tokenizer=tokenizer,
                                   training=False)
    # sampler
    batch_size = cfg['OPTIMIZATION']['BATCH_SIZE']
    trn_sampler = DistributedBatchSampler(dataset=trn_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
    dev_sampler = BatchSampler(dataset=dev_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               drop_last=False)
    # dataloader
    trn_dataloader = DataLoader(trn_dataset,
                                batch_sampler=trn_sampler,
                                num_workers=cfg['MISC']['NUM_WORKERS'])
    dev_dataloader = DataLoader(dev_dataset,
                                batch_sampler=dev_sampler,
                                num_workers=cfg['MISC']['NUM_WORKERS'])

    # 3. Build model
    config = BertConfig.from_json_file(
        os.path.join(cfg['PRETRAINED']['DIR'], 'config.json'))
    config.num_labels = cfg['OUTPUT']['NUM_LABELS']
    config.loss_type  = cfg['OPTIMIZATION']['LOSS_TYPE']
    config.img_feat_dim  = cfg['INPUT']['IMG_FEATURE_DIM']
    config.img_feat_type = cfg['INPUT']['IMG_FEATURE_TYPE']
    model = OscarForVLTaks(config=config)
    model.set_state_dict(paddle.load(
        os.path.join(cfg['PRETRAINED']['DIR'], 'paddle_model.bin')))
    logger.info('Load pretrained weights: {}'.format(cfg['PRETRAINED']['DIR']))

    num_params = sum([np.prod(param.shape) for param in model.parameters()])
    logger.info('Total parameters: %.2f M.' % (num_params / 1e6))

    # 4. Resume training or load pretrained model
    resume_epoch = 0
    if cfg['PRETRAINED']['RESUME']:
        logger.info('Resume training from {}'.format(cfg['PRETRAINED']['RESUME']))
        checkpoint = paddle.load(cfg['PRETRAINED']['RESUME'], map_location='cpu')
        model.set_state_dict(checkpoint['model'])
        resume_epoch = checkpoint['epoch'] + 1

    if args.n_gpus > 1:
        model = paddle.DataParallel(model)

    # 5. Set up optimizer & lr scheduler
    num_iters = len(trn_dataloader)
    epochs = cfg['OPTIMIZATION']['EPOCHS']
    batch_size = cfg['OPTIMIZATION']['BATCH_SIZE'] * args.n_gpus
    grad_accum_steps = cfg['OPTIMIZATION']['GRADIENT_ACCUMULATION_STEPS']
    train_optimization_steps = num_iters * (epochs - resume_epoch) // grad_accum_steps

    scheduler = set_scheduler(train_optimization_steps, cfg)
    optimizer = set_optimizer(model, scheduler, cfg)

    # 6. Training
    logger.info("** ** ** Running training ** ** **")
    logger.info("Num Iters: %d" % num_iters)
    logger.info("Batch Size: %d" % batch_size)
    logger.info('Accum Steps: %d' % grad_accum_steps)
    logger.info("Optim Steps: %d" % train_optimization_steps)

    for epoch in range(resume_epoch, epochs):
        # Train one epoch
        train_epoch(model, trn_dataloader, optimizer, scheduler, epoch, logger, args, cfg)

        # Perform evaluation
        if (epoch + 1) % cfg['MONITOR']['EVAL_FREQ'] == 0:
            validate(model, dev_dataloader, epoch, logger)

        # if (epoch + 1) % cfg['OUTPUT']['SAVE_FREQ'] == 0:
        if (epoch + 1) >= 25:
            checkpoint_dir = os.path.join(save_dir, 'checkpoint-{}'.format(epoch + 1))
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'paddle_model.bin'.format(epoch + 1))
            paddle.save({'epoch': epoch,
                         'model': model.state_dict()}, checkpoint_path)
            config.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            logger.info('** * ** Saving trained model to {}. ** * **'.format(checkpoint_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    args = parser.parse_args()

    # get default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # Set cuda device
    paddle.set_device('gpu')

    args.n_gpus = paddle.distributed.get_world_size()
    # Distributed training
    if args.n_gpus > 1:
        paddle.distributed.init_parallel_env()

    main(args, cfg)
