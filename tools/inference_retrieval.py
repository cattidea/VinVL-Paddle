#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, '.')
import argparse
from PIL import Image
import matplotlib.pyplot as plt

# utils
from utils.inference import inference


def main(args):
    i2t_results, t2i_results = inference(args.cfg_file, args.checkpoint_dir)
    if args.query_img:
        query_img_id = int(args.query_img[-10:-4])
        assert query_img_id in i2t_results, \
            "Please make sure query img in the data/minitest_images/"

        print('查询图像:', args.query_img)
        print('检索结果:')
        query_result = i2t_results[query_img_id]
        for i in range(5):
            print(f'Top{i+1}: {query_result[i]}')

    if args.query_txt:
        assert args.query_txt in t2i_results, \
            "Please make sure query txt in the data/minmitest_captions.txt"

        print('查询文本:')
        print(args.query_txt)
        print('检索结果:')
        query_result = t2i_results[args.query_txt]
        for i in range(5):
            print('Top[%d]: data/minitest_images/COCO_val2014_000000%06d.jpg' % (i+1, query_result[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str,
        default='configs/inference.yaml',
        help='Path to the config file for a specific experiment.')
    parser.add_argument('--query_img', type=str,
        help='Which img to be queried.')
    parser.add_argument('--query_txt', type=str,
        help='Which txt to be queried.')
    parser.add_argument('--checkpoint_dir', type=str,
        default='exp/finetune_retrieval_22Y_02M_13D_23H/checkpoint-30',
        help='Path to the pretrained weights.')
    args = parser.parse_args()

    # Make sure query img or txt exists
    assert args.query_img or args.query_txt, \
        "Please specify query img or query txt."

    # Make sure checkpoint dir exists
    assert os.path.isdir(args.checkpoint_dir), \
        "Please make sure the specified checkpoint dir and eval epoch exist."

    # Call main
    main(args)
