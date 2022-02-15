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
from utils.utils import compute_ranks


def main(args, cfg):
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
    config = BertConfig.from_json_file(os.path.join(args.checkpoint_dir, 'config.json'))
    config.num_labels = cfg['OUTPUT']['NUM_LABELS']
    config.loss_type  = cfg['OPTIMIZATION']['LOSS_TYPE']
    config.img_feat_dim  = cfg['INPUT']['IMG_FEATURE_DIM']
    config.img_feat_type = cfg['INPUT']['IMG_FEATURE_TYPE']
    model = OscarForVLTaks(config=config)
    checkpoint = paddle.load(os.path.join(args.checkpoint_dir, 'paddle_model.bin'))
    model.set_state_dict(checkpoint['model'])
    print('Load state dict from %s.' % args.checkpoint_dir)
    model.eval()

    # 3. Start to inference
    inference_file = os.path.join(args.checkpoint_dir, 'inference_results.pd')
    if os.path.isfile(inference_file):
        print('Found inference file in {}, skip inference.'.format(inference_file))
        results = paddle.load(inference_file)
    else:
        print('Found no inference file in {}, start to inference'.format(args.checkpoint_dir))
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
        print('Inference Done! Saving inference results to {}.'.format(inference_file))
        paddle.save(results, inference_file)

    # 4. Start to evaluate
    i2t_ranks, t2i_ranks = compute_ranks(test_dataset, results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    print("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
           i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
    print("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
           t2i_accs[0], t2i_accs[1], t2i_accs[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, required=True,
        help='Path to the config file for a specific experiment.')
    args = parser.parse_args()

    # Get the default config & merge from cfg_file
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # Make sure checkpoint dir exists
    args.checkpoint_dir = cfg['EVAL']['CHECKPOINT_DIR']
    assert os.path.isdir(args.checkpoint_dir), \
        "Please make sure the specified checkpoint dir and eval epoch exist."

    # Call main
    main(args, cfg)