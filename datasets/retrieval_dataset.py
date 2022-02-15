#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import base64
import random
import numpy as np

# torch
import paddle
from paddle.io import Dataset
# utils
from utils.io_utils import TSVFile


class RetrievalDataset(Dataset):
    """"Paddle wrapper of the image-text retrieval dataset.

    Args:
        split (str): Split of the dataset.
        cfg (yacs.config.CfgNode): A yacs.config.CfgNode object, which contains
            configuration for this experiment.
        tokenizer (BertTokenizer): A BertTokenizer object to process caption.

    """
    def __init__(self, split, cfg, tokenizer, training=True):# {{{
        super(RetrievalDataset, self).__init__()
        self.data_dir = cfg['DATASET']['DATA_DIR']
        # Self.feat_file is a tsv file,
        # which will loaded with TSVFile.
        self.feat_file = cfg['DATASET']['FEAT_FILE']
        self.feat_tsv = TSVFile(self.feat_file)
        # Self.captions is a dictionary,
        # whose key is image id and value is list of captions.
        self.text_file = os.path.join(self.data_dir, '{}_captions.pd'.format(split))
        self.captions = paddle.load(self.text_file)

        self.image_keys = list(self.captions.keys())
        if not type(self.captions[self.image_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.image_keys}

        # Get the mapping from image_id to index
        imageid2idx_file = os.path.join(os.path.dirname(self.feat_file),
            'imageid2idx.json')
        self.imageid2idx = json.load(open(imageid2idx_file))

        if cfg['INPUT']['ADD_OD_LABEL']:
            label_data_dir = os.path.dirname(self.feat_file)
            label_file = os.path.join(label_data_dir, 'predictions.tsv')
            self.label_tsv = TSVFile(label_file)
            self.labels = {}
            for line_num in range(self.label_tsv.num_rows()):
                row = self.label_tsv.seek(line_num)
                image_id = row[0]
                if int(image_id) in self.image_keys:
                    results = json.loads(row[1])
                    objects = results['objects'] if type(
                        results) == dict else results
                    self.labels[int(image_id)] = {
                        'image_h': results['image_h'] if type(
                            results) == dict else 600,
                        'image_w': results['image_w'] if type(
                            results) == dict else 800,
                        'class': [cur_d['class'] for cur_d in objects],
                        'boxes': np.array([cur_d['rect'] for cur_d in objects],
                                          dtype=np.float32)
                    }
            self.label_tsv._fp.close()
            self.label_tsv._fp = None

        if training:
            self.num_captions_per_image = cfg['INPUT']['NUM_CAPTIONS_PER_IMAGE_TRN']
        else:
            self.num_captions_per_image = cfg['INPUT']['NUM_CAPTIONS_PER_IMAGE_DEV']

            if cfg['EVAL']['EVAL_IMG_KEYS_FILE']:
                # Select a subset of image keys for evaluation.
                # E.g., COCO 1K and 5K.
                with open(os.path.join(self.data_dir, cfg['EVAL']['EVAL_IMG_KEYS_FILE']), 'r') as f:
                    image_keys = f.readlines()
                self.image_keys = [int(k.strip()) for k in image_keys]
                self.captions = {k: self.captions[k] for k in self.image_keys}
                if cfg['INPUT']['ADD_OD_LABEL']:
                    self.labels = {k: self.labels[k] for k in self.image_keys}

            if cfg['EVAL']['EVAL_CAPTION_INDEX_FILE']:
                # Hard negative image/caption indexs for retrieval re-rank setting.
                # Useful for mini val set to monitor the performance during training.
                # However, it cannot be used together with cross image evaluation.
                self.has_caption_indices = True
                assert not cfg['EVAL']['EVAL_CROSS_IMAGE']
                caption_index_file = os.path.join(self.data_dir,
                    cfg['EVAL']['EVAL_CAPTION_INDEX_FILE'])
                self.caption_indices = paddle.load(caption_index_file)
                if not type(self.caption_indices[self.image_keys[0]]) == list:
                    self.caption_indices = {k: json.loads(self.caption_indices[k])
                                            for k in self.image_keys}
            else:
                self.has_caption_indices = False

        self.cfg = cfg
        self.training = training
        self.tokenizer = tokenizer
        self.max_seq_len = cfg['INPUT']['MAX_SEQ_LEN']
        self.max_img_seq_len = cfg['INPUT']['MAX_REGION']# }}}


    def get_image_caption_index(self, index):# {{{
        """Return img_idx to access features and [img_key, cap_idx] to access 
        caption.

        Args:
            index (int): The index-th example.

        Returns:
            ~(Int, List):

            * **img_idx**: The img_idx-th image in the dataset of the index-th
                example. This value is used to access feature of the image corrsponds
                to this example.
            * **cap_idxs**: A List consists of two elements. The element is the image
                id of this example, and the second element indicates which caption 
                corrsponds to this example.

        """
        if not self.training and self.cfg['EVAL']['EVAL_CROSS_IMAGE']:
            img_idx = index // (self.num_captions_per_image * len(self.image_keys))
            cap_idx = index % (self.num_captions_per_image * len(self.image_keys))
            img_idx1 = cap_idx // self.num_captions_per_image
            cap_idx1 = cap_idx % self.num_captions_per_image
            return img_idx, [self.image_keys[img_idx1], cap_idx1]

        if not self.training and self.has_caption_indices:
            img_idx = index // self.num_captions_per_image
            cap_idx = index % self.num_captions_per_image
            img_key1, cap_idx1 = \
                    self.caption_indices[self.image_keys[img_idx]][cap_idx]
            return img_idx, [img_key1, cap_idx1]

        img_idx = index // self.num_captions_per_image
        cap_idx = index % self.num_captions_per_image
        return img_idx, [self.image_keys[img_idx], cap_idx]# }}}


    def get_feat(self, image_id):# {{{
        image_idx = self.imageid2idx[str(image_id)]
        row = self.feat_tsv.seek(image_idx)
        num_boxes = int(row[1])
        features = np.frombuffer(base64.b64decode(row[-1]),
                                 dtype=np.float32).reshape((num_boxes, -1))
        feat = paddle.to_tensor(features)
        return feat# }}}


    def get_label(self, index):# {{{
        img_idx, cap_idx = self.get_image_caption_index(index)
        return 1 if self.image_keys[img_idx] == cap_idx[0] else 0# }}}


    def _get_od_label(self, image_id):# {{{
        """Fetch object detection labels of the given image.

        Args:
            image_id (int): The id of the image, e.g., .

        Returns:
            ~str:

            * **od_label**: Concatenate all the detected object labels into a string.

        """
        if self.cfg['INPUT']['ADD_OD_LABEL']:
            if type(self.labels[image_id]) == str:
                od_label = self.labels[image_id]
            else:
                od_label = ' '.join(self.labels[image_id]['class'])
            return od_label# }}}

    
    def _tensorize_example(self, text_a, feature, text_b=None,# {{{
                           cls_token_segment_id=0, pad_token_segment_id=0,
                           sequence_a_segment_id=0, sequence_b_segment_id=1):
        # Tokenize text_a (caption) using the tokenizer
        tokens_a = self.tokenizer.tokenize(text_a)
        if len(tokens_a) > self.max_seq_len - 2:
            tokens_a = tokens_a[:(self.max_seq_len - 2)]
        # [CLS] + tokens_a + [SEP]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [cls_token_segment_id] + [sequence_a_segment_id] * (len(tokens_a) + 1)
        
        # Tokenize text_b (od_label)
        if text_b:
            tokens_b = self.tokenizer.tokenize(text_b)
            if len(tokens_b) > self.max_seq_len - len(tokens) - 1:
                tokens_b = tokens_b[: (self.max_seq_len - len(tokens) -1)]
            # tokens_b + [SEP]
            tokens += tokens_b + [self.tokenizer.sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        # Padding
        seq_len = len(tokens)
        seq_padding_len = self.max_seq_len - seq_len
        tokens += [self.tokenizer.pad_token] * seq_padding_len
        segment_ids += [pad_token_segment_id] * seq_padding_len
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Tensorize feature
        img_len = feature.shape[0]
        if img_len > self.max_img_seq_len:
            feature = feature[0: self.max_img_seq_len, :]
            img_len = feature.shape[0]
            img_padding_len = 0
        else:
            img_padding_len = self.max_img_seq_len - img_len
            padding_matrix = paddle.zeros((img_padding_len, feature.shape[1]))
            feature = paddle.concat((feature, padding_matrix), axis=0)

        # Generate attention mask
        att_mask_type = self.cfg['INPUT']['ATT_MASK_TYPE']
        assert att_mask_type == 'CLR'
        attention_mask = [1] * seq_len + [0] * seq_padding_len + \
                         [1] * img_len + [0] * img_padding_len
        
        # Construct inputs
        input_ids = paddle.to_tensor(input_ids, dtype=paddle.int64)
        attention_mask = paddle.to_tensor(attention_mask, dtype=paddle.int64)
        segment_ids = paddle.to_tensor(segment_ids, dtype=paddle.int64)
        return (input_ids, attention_mask, segment_ids, feature)# }}}


    def __len__(self):# {{{
        if not self.training and self.cfg['EVAL']['EVAL_CROSS_IMAGE']:
            return len(self.image_keys) ** 2 * self.num_captions_per_image
        return len(self.image_keys) * self.num_captions_per_image# }}}


    def __getitem__(self, index):# {{{
        if self.training:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.image_keys[img_idx]  # image id
            feature = self.get_feat(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_label = self._get_od_label(img_key)
            example = self._tensorize_example(caption, feature, text_b=od_label)

            # Select a negative pair
            neg_img_indexs = list(range(0, img_idx)) + \
                    list(range(img_idx + 1, len(self.image_keys)))
            neg_img_idx = random.choice(neg_img_indexs)
            if random.random() <= 0.5:
                # Randomly select a negative caption from a different image, i.e., neg caption.
                neg_cap_idx = random.randint(0, self.num_captions_per_image - 1)
                neg_caption = self.captions[self.image_keys[neg_img_idx]][neg_cap_idx]
                neg_example = self._tensorize_example(neg_caption, feature, text_b=od_label)
            else:
                # Randomly select a negative image, i.e., neg image.
                neg_feature = self.get_feat(self.image_keys[neg_img_idx])
                neg_od_label = self._get_od_label(self.image_keys[neg_img_idx])
                neg_example = self._tensorize_example(caption, neg_feature, text_b=neg_od_label)

            example_pair = tuple(list(example) + [1] + list(neg_example) + [0])
            return index, example_pair
        else:
            img_idx, cap_idxs = self.get_image_caption_index(index)
            img_key = self.image_keys[img_idx]  # image id
            feature = self.get_feat(img_key)
            caption = self.captions[cap_idxs[0]][cap_idxs[1]]
            od_label = self._get_od_label(img_key)
            example = self._tensorize_example(caption, feature, text_b=od_label)
            label = 1 if img_key == cap_idxs[0] else 0
            return index, tuple(list(example) + [label])# }}}
