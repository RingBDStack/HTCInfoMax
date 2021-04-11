#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch
import helper.logger as logger
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
import re

INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ):
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])
        self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index)

        all_label_words = []
        self.label_map_splittedwords = dict()
        for label_index in vocab_map.keys():
            each_label = self.clean_str(vocab_map[label_index])
            each_label_list = [word.lower() for word in each_label.split() if len(word) > 1]
            all_label_words += each_label_list
            self.label_map_splittedwords[label_index] = each_label_list
        self.final_lookup_table = torch.zeros(len(vocab_map), embedding_dim)
        all_label_words_dict = dict()
        index = 0
        for ind, word in enumerate(all_label_words):
            if word not in all_label_words_dict.keys():
                all_label_words_dict[word] = index
                index += 1

        # initialize lookup table
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(all_label_words_dict),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(all_label_words_dict),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(all_label_words_dict),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25)

        if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain' \
                and pretrained_dir is not None and pretrained_dir != '':
            self.load_pretrained(embedding_dim, all_label_words_dict, vocab_name, pretrained_dir)

        if padding_index is not None:
            self.lookup_table[padding_index] = 0.0

        for label in self.label_map_splittedwords.keys():
            num_words = len(self.label_map_splittedwords[label])
            each_label_embedding = torch.zeros(num_words, embedding_dim)
            for ind, eachword in enumerate(self.label_map_splittedwords[label]):
                each_label_embedding[ind] = self.lookup_table[all_label_words_dict[eachword]]
            each_label_embedding = torch.mean(each_label_embedding, 0)
            self.final_lookup_table[label] = each_label_embedding

        self.embedding.weight.data.copy_(self.final_lookup_table)
        self.embedding.weight.requires_grad = True
        del self.final_lookup_table

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
            embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0
            for line in f_in:
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), embedding_dim)
                    continue
                if row[0] in vocab_map:
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[vocab_map[row[0]]] = current_embedding
                    num_pretrained_vocab += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = string.strip().strip('"')
        string = re.sub(r"[^A-Za-z0-9(),!?\.\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " ", string)
        string = re.sub(r"\.", " ", string)
        string = re.sub(r"\"", " ", string)
        string = re.sub(r"!", " ", string)
        string = re.sub(r"\(", " ", string)
        string = re.sub(r"\)", " ", string)
        string = re.sub(r"\?", " ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)
