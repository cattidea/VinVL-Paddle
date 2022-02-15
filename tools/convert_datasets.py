#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import paddle
import numpy as np

paddle_data_dir = '/mnt/disk2T/Data/Research/Multi-Modal-Pretraining/2021-VinVL-CVPR/coco_ir_paddle'
pytorch_data_dir = '/mnt/disk2T/Data/Research/Multi-Modal-Pretraining/2021-VinVL-CVPR/coco_ir'

pytorch_trn_text_file = os.path.join(pytorch_data_dir, 'train_captions.pt')
pytorch_dev_text_file = os.path.join(pytorch_data_dir, 'val_captions.pt')
pytorch_test_text_file = os.path.join(pytorch_data_dir, 'test_captions.pt')
pytorch_minidev_text_file = os.path.join(pytorch_data_dir, 'minival_captions.pt')

paddle_trn_text_file = os.path.join(paddle_data_dir, 'train_captions.pd')
paddle_dev_text_file = os.path.join(paddle_data_dir, 'val_captions.pd')
paddle_test_text_file = os.path.join(paddle_data_dir, 'test_captions.pd')
paddle_minidev_text_file = os.path.join(paddle_data_dir, 'minival_captions.pd')

pytorch_minidev_cap_index_file = os.path.join(pytorch_data_dir, 'minival_caption_indexs_top20.pt')
paddle_minidev_cap_index_file = os.path.join(paddle_data_dir, 'minival_caption_indexs_top20.pd')


###########################################################
################# Convert caption files ###################
###########################################################
process_paddle_text_file = paddle_trn_text_file
process_pytorch_text_file = pytorch_trn_text_file

paddle_text = {}
pytorch_text = torch.load(process_pytorch_text_file)
for k, v in pytorch_text.items():
    paddle_text[k] = json.loads(v)

paddle.save(paddle_text, process_paddle_text_file)

#################################################################
################# Convert caption index file ####################
#################################################################
# paddle_cap_index = {}
# pytorch_cap_index = torch.load(pytorch_minidev_cap_index_file)

# for k, v in pytorch_cap_index.items():
#     paddle_cap_index[k] = json.loads(v)

# paddle.save(paddle_cap_index, paddle_minidev_cap_index_file)


