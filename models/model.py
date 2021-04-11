#!/usr/bin/env python
# coding:utf-8

import torch.nn as nn
import torch
import numpy as np
from models.structure_model.structure_encoder import StructureEncoder
from models.text_encoder import TextEncoder
from models.embedding_layer import EmbeddingLayer
from models.multi_label_attention import HiAGMLA

from models.text_label_mi_discriminator import TextLabelMIDiscriminator

from models.labelprior_discriminator import LabelPriorDiscriminator
import torch.nn.functional as F



class HTCInfoMax(nn.Module):
    def __init__(self, config, vocab, model_mode='TRAIN'):
        """
        HTCInfoMax Model class
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param model_mode: Str, ('TRAIN', 'EVAL'), initialize with the pretrained word embedding if value is 'TRAIN'
        """
        super(HTCInfoMax, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.train.device_setting.device

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.index2label = vocab.i2v['label']

        self.token_embedding = EmbeddingLayer(
            vocab_map=self.token_map,
            embedding_dim=config.embedding.token.dimension,
            vocab_name='token',
            config=config,
            padding_index=vocab.padding_index,
            pretrained_dir=config.embedding.token.pretrained_file,
            model_mode=model_mode,
            initial_type=config.embedding.token.init_type
        )

        # linear layer used for learning the weights for text_label_mi loss and label_prior_matching loss
        self.labelpriorweight_linear = nn.Linear(len(self.label_map) * config.embedding.label.dimension, 1)
        self.text_label_MI_weight_linear = nn.Linear(config.embedding.label.dimension, 1)

        self.text_encoder = TextEncoder(config)
        self.structure_encoder = StructureEncoder(config=config,
                                                  label_map=vocab.v2i['label'],
                                                  device=self.device,
                                                  graph_model_type=config.structure_encoder.type)

        self.label_prior_d = LabelPriorDiscriminator()
        self.text_label_mi_d = TextLabelMIDiscriminator()

        self.htcinfomax = HiAGMLA(config=config,
                                 device=self.device,
                                 graph_model=self.structure_encoder,
                                 label_map=self.index2label,
                                 model_mode=model_mode)
        
    def optimize_params_dict(self):
        """
        get parameters of the overall model
        :return: List[Dict{'params': Iteration[torch.Tensor],
                           'lr': Float (predefined learning rate for specified module,
                                        which is different from the others)
                          }]
        """
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.token_embedding.parameters()})
        params.append({'params': self.htcinfomax.parameters()})
        return params

    def forward(self, batch):
        """
        forward pass of the overall architecture
        :param batch: DataLoader._DataLoaderIter[Dict{'token_len': List}], each batch sampled from the current epoch
        :return: 
        """

        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        embedding = self.token_embedding(batch['token'].to(self.config.train.device_setting.device))

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']
        token_output = self.text_encoder(embedding, seq_len)

        all_labels_feature, logits = self.htcinfomax(token_output)

        text_feature = token_output
        idx = np.random.permutation(text_feature.shape[0])
        negative_text = text_feature[idx, :, :]

        for i, label_index in enumerate(batch['label_list']):
            # Label Selector: select the corresponding labels for each text sample
            label_feature = all_labels_feature[label_index,:]
            label_feature_mean = torch.mean(label_feature, dim=0, keepdim=True)
            if i == 0:
                label_feature_y = label_feature_mean
            else:
                label_feature_y = torch.cat((label_feature_y, label_feature_mean), dim=0)
                

        # compute the text-label mutual information maximization loss
        t = text_feature.permute(0, 2, 1)
        t_prime = negative_text.permute(0, 2, 1)
        E_joint = -F.softplus(-self.text_label_mi_d(label_feature_y, t)).mean()
        E_marginal = F.softplus(self.text_label_mi_d(label_feature_y, t_prime)).mean()
        text_label_mi_disc_loss = (E_marginal - E_joint)


        # compute the label prior matching loss
        label_totalnum = all_labels_feature.shape[0]
        label_prior_loss = 0.0
        for i in range(label_totalnum):
            label_y = all_labels_feature[i]
            label_prior = torch.rand_like(label_y)
            term_a = torch.log(self.label_prior_d(label_prior)).mean()
            term_b = torch.log(1.0 - self.label_prior_d(label_y)).mean()
            label_prior_loss += - (term_a + term_b)
        label_prior_loss /= label_totalnum

        
        # loss weight estimator: compute the weights for above two losses
        text_feature_temp = torch.mean(text_feature, dim=1)
        text_feature_mean = torch.mean(text_feature_temp, dim=0)
        text_label_MI_weightlogit = self.text_label_MI_weight_linear(text_feature_mean)
        labelprior_weightlogit = self.labelpriorweight_linear(all_labels_feature.view(-1))
        fusiongate = F.sigmoid(text_label_MI_weightlogit + labelprior_weightlogit)

        return text_label_mi_disc_loss, label_prior_loss, logits, fusiongate
