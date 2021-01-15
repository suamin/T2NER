# -*- coding: utf-8 -*-

import os
import logging
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

IGNORE_INDEX = torch.nn.CrossEntropyLoss().ignore_index


class TokenizerFeatures(ABC):
    
    def __init__(self, **kwargs):
        # By default we consider using mBERT
        kwargs = dict(
            tokenizer_name_or_path=kwargs.get(
                "tokenizer_name_or_path", 
                "bert-base-multilingual-cased"
            ),
            do_lower_case=kwargs.get("do_lower_case", False),
            max_seq_length=kwargs.get("max_seq_length", 128),
            cache_dir=kwargs.get("cache_dir", None)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            kwargs["tokenizer_name_or_path"],
            do_lower_case=kwargs["do_lower_case"],
            cache_dir=kwargs["cache_dir"]
        )
        self.kwargs = kwargs
    
    @property
    def pad_token(self):
        return self.tokenizer.pad_token
    
    @property
    def pad_token_input_id(self):
        return self.tokenizer.pad_token_id
    
    @property
    def pad_token_label_id(self):
        return IGNORE_INDEX
    
    @property
    def max_seq_length(self):
        return self.kwargs["max_seq_length"]
    
    @abstractmethod
    def convert_example_to_feature(self, example, verbose=False):
        pass
    
    def convert_examples_to_features(self, examples):
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            if ex_index < 5:
                verbose = True
            else:
                verbose = False
            features.append(self.convert_example_to_feature(example, verbose))
        return features
    
    @staticmethod
    def save_features(features, cached_features_file):
        logger.info("Saving features into cached file {}".format(cached_features_file))
        torch.save(features, cached_features_file)
    
    @staticmethod
    def load_features(cached_features_file):
        logger.info("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
        return features


def read_conll_ner_file(file_path):
    examples = list()
    
    if not os.path.exists(file_path):
        logger.warn("File {} not exists".format(file_path))
    else:
        with open(file_path, encoding="utf-8", errors="ignore") as rf:
            words = list()
            labels = list()
            
            for line in rf.readlines():
                line = line.strip()
                if line.startswith("-DOCSTART-") or line == "":
                    if words:
                        examples.append({"words": words, "labels": labels})
                        words = list()
                        labels = list()
                else:
                    split = line.split("\t")
                    word = split[0]
                    words.append(split[0])
                    
                    if len(split) > 1:
                        label = split[-1].replace("\n", "")
                        labels.append(label)
                    else:
                        # Possibly no label in test mode
                        labels.append("O")
            
            if words:
                examples.append({"words": words, "labels": labels})
    
    return examples


def read_labels(file_path):
    labels = set()
    with open(file_path) as rf:
        for line in rf:
            line = line.strip()
            if line:
                labels.add(line)
    if "O" not in labels:
        labels.add("O")
    labels = list(sorted(labels))
    return labels
