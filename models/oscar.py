#! /usr/bin/env python
# -*- coding: utf-8 -*-

import math
import logging

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import LayerNorm as BertLayerNorm

# bert modules
from models.bert import (
    BertLayer,
    BertOutput,
    BertPooler,
    BertEncoder,
    BertAttention,
    BertEmbeddings,
    BertSelfOutput,
    BertIntermediate,
    BertSelfAttention,
    BertPreTrainedModel,
)

class CaptionBertSelfAttention(BertSelfAttention):# {{{
    """Modified from BertSelfAttention to add support for output_hidden_states."""
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        if history_state is not None:
            x_states = paddle.concat([history_state, hidden_states], axis=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = paddle.matmul(query_layer, key_layer.transpose((0, 1, 3, 2)))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(axis=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = paddle.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose((0, 2, 1, 3))
        new_context_layer_shape = context_layer.shape[:-2] + [self.all_head_size, ]
        context_layer = context_layer.reshape(shape=new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs# }}}


class CaptionBertAttention(BertAttention):# {{{
    """Modified from BertAttention to add support for output_hidden_states."""
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, head_mask=None,
                history_state=None):
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output, ) + self_outputs[1:]  # Add attentions if we output them
        return outputs# }}}


class CaptionBertLayer(BertLayer):# {{{
    """Modified from BertLayer to add support for output_hidden_states."""
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[1:]  # Add attention is we output them
        return outputs# }}}


class CaptionBertEncoder(BertEncoder):# {{{
    """Modified from BertEncoder to add support for output_hidden_states."""
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.LayerList([CaptionBertLayer(config)
                                   for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            history_state = None if encoder_history_states is None \
                    else encoder_history_states[i]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs  # outputs, (hidden_states), (attentions)}}}


class ImgBertModel(BertPreTrainedModel):# {{{
    """Modified from BertModel to handle image region features as input."""
    def __init__(self, config):
        super(ImgBertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = CaptionBertEncoder(config)
        self.pooler = BertPooler(config)

        self.img_feat_dim = config.img_feat_dim
        self.img_feat_type = config.img_feat_type
        if hasattr(config, 'use_img_layernorm'):
            self.use_img_layernorm = config.use_img_layernorm
        else:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_feat_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, epsilon=config.img_layer_norm_eps)

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                position_ids=None, head_mask=None, img_feats=None,
                encoder_history_states=None):
        if attention_mask is None:
            attention_mask = paddle.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch, 1, 1, to_seq_length]
        # So we can broadcast to [batch, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPI, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.cast(dtype=paddle.float32)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape
        # [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to float if needed + fp16 compatibility
            head_mask = head_mask.cast(dtype=paddle.float32)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids)

        if encoder_history_states:
            assert img_feats is None, \
                    "Cannot take image features while using encoder history states."

        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)
            if self.use_img_layernorm:
                img_embedding_output = self.LayerNorm(img_embedding_output)

            # Add dropout on image embedding
            img_embedding_output = self.dropout(img_embedding_output)
            # Concatenate two embeddings
            embedding_output = paddle.concat((embedding_output, img_embedding_output), axis=1)

        encoder_outputs = self.encoder(embedding_output, extended_attention_mask,
                head_mask=head_mask, encoder_history_states=encoder_history_states)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # Add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, ) + encoder_outputs[1:]
        return outputs# }}}


class OscarForVLTaks(BertPreTrainedModel):
    def __init__(self, config):
        super(OscarForVLTaks, self).__init__(config)
        self.config = config
        self.loss_type = config.loss_type
        self.num_labels = config.num_labels
        self.bert = ImgBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if hasattr(config, 'classifier'):
            if not hasattr(config, 'cls_hidden_scale'):
                config.cls_hidden_scale = 2

            if config.classifier == 'linear':
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            elif config.classifier == 'mlp':
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size * config.cls_hidden_scale),
                    nn.ReLU(),
                    nn.Linear(config.hidden_size * config.cls_hidden_scale, config.num_labels)
                )
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, img_feats=None):
        outputs =self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                           attention_mask=attention_mask, head_mask=head_mask, img_feats=img_feats)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Add hidden states and attention if they are here
        outputs = (logits, ) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:  # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.reshape((-1, )), labels.reshape((-1, )))
            else:
                if self.loss_type == 'sfmx':
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(input=logits.reshape((-1, self.num_labels)),
                                    label=labels.reshape((-1, )))
            outputs = (loss, ) + outputs
        return outputs