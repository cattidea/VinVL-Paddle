#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '.')
import argparse
from tqdm import tqdm

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from paddlenlp.transformers.bert.tokenizer import BertTokenizer

# model
from models.bert import BertConfig
from models.oscar import OscarForVLTaks
# dataset
from datasets.retrieval_dataset import RetrievalDataset
# config
from config.default import get_cfg_defaults
# utils
from utils.utils import get_retrieval_results 


def inference(cfg_file, checkpoint_dir):
    # 0. Preparation
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)

    # 1. Create test dataloader
    tokenizer = BertTokenizer.from_pretrained(cfg['INPUT']['BERT_MODEL'])
    test_dataset = RetrievalDataset(split=cfg['DATASET']['TEST'],
                                    cfg=cfg,
                                    tokenizer=tokenizer,
                                    training=False)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=cfg['OPTIMIZATION']['BATCH_SIZE'],
                                 num_workers=cfg['MISC']['NUM_WORKERS'],
                                 drop_last=False)

    # 2. Build model
    config = BertConfig.from_json_file(os.path.join(checkpoint_dir, 'config.json'))
    config.num_labels = cfg['OUTPUT']['NUM_LABELS']
    config.loss_type  = cfg['OPTIMIZATION']['LOSS_TYPE']
    config.img_feat_dim  = cfg['INPUT']['IMG_FEATURE_DIM']
    config.img_feat_type = cfg['INPUT']['IMG_FEATURE_TYPE']
    model = OscarForVLTaks(config=config)
    checkpoint = paddle.load(os.path.join(checkpoint_dir, 'paddle_model.bin'))
    model.set_state_dict(checkpoint['model'])
    print('Load state dict from %s.' % checkpoint_dir)
    model.eval()

    # 3. Start to inference
    results = {}
    for inds, batch in tqdm(test_dataloader):
        with paddle.no_grad():
            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'token_type_ids':  batch[2],
                'img_feats':       batch[3],
                'labels':          batch[4],
            }
            _, logits = model(**inputs)[:2]
            probs = F.softmax(logits, axis=1)
            # The confidence to be a matched pair
            result = probs[:, 1]
            inds = [inds[i].item() for i in range(inds.shape[0])]
            result = [result[i].item() for i in range(result.shape[0])]
            results.update({ind: res for ind, res in zip(inds, result)})

    # 4. Start to evaluate
    i2t_results, t2i_results = get_retrieval_results(test_dataset, results)
    return i2t_results, t2i_results

