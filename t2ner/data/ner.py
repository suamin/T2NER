# -*- coding: utf-8 -*-

import os
import logging
import torch
import random
import collections
import functools

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import XLMTokenizer

from . import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample:
    
    def __init__(
        self,
        guid,
        words,
        labels=None,
        lang=None,
        domain=None,
        dataset_source=None
    ):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.lang = lang
        self.domain = domain
        self.dataset_source = dataset_source


class InputFeatures:
    
    def __init__(
        self,
        input_ids,
        input_mask=None,
        label_ids=None,
        type_ids=None,
        all_outside_id=None,
        lang_id=None,
        domain_id=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.type_ids = type_ids
        self.all_outside_id = all_outside_id
        self.lang_id = lang_id
        self.domain_id = domain_id


class NERFeatures(utils.TokenizerFeatures):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_list = sorted(list(set(kwargs.pop("label_list", ["O"]))))
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.type_list = sorted(list(set(kwargs.pop("type_list", ["O"]))))
        self.type2id = {etype: idx for idx, etype in enumerate(self.type_list)}
    
    @property
    def id2label(self):
        return {idx: label for label, idx in self.label2id.items()}
    
    @property
    def id2type(self):
        return {idx: etype for etype, idx in self.type2id.items()}
    
    def encode(self, input_ids, label_ids, type_ids):
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        label_ids = self.tokenizer.build_inputs_with_special_tokens(label_ids)
        type_ids = self.tokenizer.build_inputs_with_special_tokens(type_ids)
        
        special_mask = (
            self.tokenizer
            .get_special_tokens_mask(input_ids, already_has_special_tokens=True)
        )
        label_ids = [
            self.pad_token_label_id if special_mask[i] == 1 else j
            for i, j in enumerate(label_ids)
        ]
        type_ids = [
            self.pad_token_label_id if special_mask[i] == 1 else j
            for i, j in enumerate(type_ids)
        ]
        
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            label_ids = label_ids[:self.max_seq_length]
            type_ids = type_ids[:self.max_seq_length]
        
        input_mask = [1] * len(input_ids)
        padding_length = self.max_seq_length - len(input_ids)
        
        if self.tokenizer.padding_side == "left":
            input_ids = ([self.pad_token_input_id] * padding_length) + input_ids
            input_mask = ([0] * padding_length) + input_mask
            label_ids = ([self.pad_token_label_id] * padding_length) + label_ids
            type_ids = ([self.pad_token_label_id] * padding_length) + type_ids
        else:
            input_ids += ([self.pad_token_input_id] * padding_length)
            input_mask += ([0] * padding_length)
            label_ids += ([self.pad_token_label_id] * padding_length)
            type_ids += ([self.pad_token_label_id] * padding_length)
        
        return input_ids, input_mask, label_ids, type_ids
    
    def convert_example_to_feature(self, example, verbose=False):
        tokens = list()
        label_ids = list()
        type_ids = list()
        all_outside = list()

        for token, label in zip(example.words, example.labels):
            if isinstance(self.tokenizer, XLMTokenizer):
                sub_tokens = self.tokenizer.tokenize(token, example.lang)
            else:
                sub_tokens = self.tokenizer.tokenize(token)
            
            if not sub_tokens:
                continue
            
            tokens.extend(sub_tokens)
            # Use the real label id for the first token of the word, and 
            # padding ids for the remaining tokens
            label_id = self.label2id[label]
            label_ids.extend(
                [label_id] + [self.pad_token_label_id] * (len(sub_tokens) - 1)
            )
            all_outside.append(label)
            # Similarly for entity types
            etype = "O" if label == "O" else label[2:]
            type_id = self.type2id[etype]
            type_ids.extend(
                [type_id] + [self.pad_token_label_id] * (len(sub_tokens) - 1)
            )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids, input_mask, label_ids, type_ids = self.encode(input_ids, label_ids, type_ids)
        all_outside = sum([label == "O" for label in all_outside]) == len(example.words)
        all_outside_id = 1 if all_outside else 0
        
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(type_ids) == self.max_seq_length
        
        if verbose:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("dataset: %s", example.dataset_source)
            logger.info("lang: %s", example.lang)
            logger.info("domain: %s", example.domain)
            
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
            text = " ".join([t for t in tokens if t != self.pad_token])
            
            logger.info("text: %s", text)
            logger.info("all_outside_id: %s", all_outside_id)
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info("type_ids: %s", " ".join([str(x) for x in type_ids]))    
        
        # As lang_id, domain_id can change depending on the number of langs and
        # domains considered in experiment, we create features without this info 
        # and add them on the fly as they are constant for one feature set
        feature = InputFeatures(input_ids, input_mask, label_ids, type_ids, all_outside_id)
        
        return feature


class NERDataset(Dataset):
    
    def __init__(
        self,
        features,
        id2label=None,
        id2type=None,
        unique_name=None
    ):  
        self.features = features
        self.id2label = id2label
        self.id2type = id2type
        self.unique_name = unique_name
        
        logger.info("  ****  No. of features  **** : {}".format(len(features)))
    
    @staticmethod
    def read_examples_from_file(
        input_file,
        guid_suffix=None,
        language=None,
        domain=None,
        dataset_source=None
    ):
        examples = list()
        for guid_index, example in enumerate(utils.read_conll_ner_file(input_file)):
            guid = guid_index + 1
            if guid_suffix is not None:
                guid = "{}-{}".format(guid, guid_suffix)
            example = InputExample(
                guid,
                example["words"],
                labels=example["labels"],
                lang=language,
                domain=domain,
                dataset_source=dataset_source
            )
            examples.append(example)
        return examples
    
    @staticmethod
    def load_and_cache_features_as_datasets(
        input_file=None,
        label_list=None,
        preprocessed_data_dir=None,
        unique_name=None,
        splits=("train", "dev", "test"),
        features_cache_dir=None,
        overwrite_cache=False,
        **features_kwargs
    ):
        if features_cache_dir is None:
            features_cache_dir = ""
        
        if not os.path.exists(features_cache_dir) and features_cache_dir:
            os.makedirs(features_cache_dir)
        
        if input_file:
            cache_suffix = os.path.splitext(os.path.split(input_file)[1])[0]
            cached_features_file = os.path.join(
                features_cache_dir, "cached_{}".format(cache_suffix)
            )
            
            if os.path.exists(cached_features_file) and not overwrite_cache:
                features, id2label, id2type = utils.TokenizerFeatures.load_features(cached_features_file)
            else:
                if label_list is not None:
                    features_kwargs["label_list"] = label_list
                    type_list = [label[2:] for label in label_list if label != "O"] + ["O"]
                    type_list = list(sorted(set(type_list)))
                    features_kwargs["type_list"] = type_list
                featurizer = NERFeatures(**features_kwargs)
                id2label = featurizer.id2label
                id2type = featurizer.id2type
                examples = NERDataset.read_examples_from_file(input_file)
                features = featurizer.convert_examples_to_features(examples)
                utils.TokenizerFeatures.save_features((features, id2label, id2type), cached_features_file)
            
            dataset = NERDataset(features, id2label=id2label, id2type=id2type)
            
            yield None, dataset
        
        else:
            if preprocessed_data_dir is None:
                raise ValueError(
                    "Missing required input preprocessed_data_dir."
                )
            
            if unique_name is None:
                raise ValueError(
                    "Missing required input unique_name. Should be a string "
                    "in format language.domain.dataset_source"
                )
            
            language, domain, dataset_source = unique_name.split(".")
            
            # When preprocess.py is used, *.labels file is automatically
            # generated so ``label_list`` is ignored and we read labels
            # from that file for `NERFeatures` class
            label_file = os.path.join(
                preprocessed_data_dir, "{}.labels".format(unique_name)
            )
            label_list = utils.read_labels(label_file)
            type_list = [label[2:] for label in label_list if label != "O"] + ["O"]
            type_list = list(sorted(set(type_list)))
            featurizer = None
            
            for split in splits:
                # Check if features were created previously, load them
                cache_suffix = "{}-{}".format(unique_name, split)
                cached_features_file = os.path.join(
                    features_cache_dir, "cached_{}".format(cache_suffix)
                )
                if os.path.exists(cached_features_file) and not overwrite_cache:
                    features, id2label, id2type = utils.TokenizerFeatures.load_features(
                        cached_features_file
                    )
                else:
                    file_path = os.path.join(
                        preprocessed_data_dir, "{}-{}".format(unique_name, split)
                    )
                    if not os.path.exists(file_path):
                        logger.warn("file is missing {}".format(file_path))
                        continue
                    examples = NERDataset.read_examples_from_file(
                        file_path,
                        guid_suffix=split,
                        language=language,
                        domain=domain,
                        dataset_source=dataset_source
                    )
                    if featurizer is None:
                        features_kwargs["label_list"] = label_list
                        features_kwargs["type_list"] = type_list
                        featurizer = NERFeatures(**features_kwargs)
                        id2label = featurizer.id2label
                        id2type = featurizer.id2type
                    features = featurizer.convert_examples_to_features(examples)
                    utils.TokenizerFeatures.save_features((features, id2label, id2type), cached_features_file)
                
                dataset = NERDataset(features, id2label=id2label, id2type=id2type, unique_name=unique_name)
                
                yield split, dataset
    
    def decode_label_ids(self, label_ids):
        return [
            self.id2label[i] 
            if i != utils.IGNORE_INDEX else utils.IGNORE_INDEX 
            for i in label_ids
        ]
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def get_class_counts(self):
        class_counts = collections.Counter()
        for f in self.features:
            class_counts.update(f.label_ids)
        if utils.IGNORE_INDEX in class_counts:
            del class_counts[utils.IGNORE_INDEX]
        class_counts = [c for _, c in sorted(class_counts.items(), key=lambda x: x[0])]
        return class_counts
    
    def get_type_counts(self):
        type_counts = collections.Counter()
        for f in self.features:
            type_counts.update(f.type_ids)
        if utils.IGNORE_INDEX in type_counts:
            del type_counts[utils.IGNORE_INDEX]
        type_counts = [c for _, c in sorted(type_counts.items(), key=lambda x: x[0])]
        return type_counts
    
    def add_metadata(self, lang_id, domain_id):
        for f in self.features:
            f.lang_id = lang_id
            f.domain_id = domain_id
    
    @staticmethod
    def collate_fn(data):
        batch = dict(
            input_ids=list(), 
            input_mask=list(), 
            label_ids=list(), 
            type_ids=list(),
            all_outside_id=list(),
            lang_id=list(), 
            domain_id=list()
        )
        
        for f in data:
            batch["input_ids"].append(torch.tensor(f.input_ids).long().unsqueeze(0))
            batch["input_mask"].append(torch.tensor(f.input_mask).long().unsqueeze(0))
            batch["label_ids"].append(torch.tensor(f.label_ids).long().unsqueeze(0))
            batch["type_ids"].append(torch.tensor(f.type_ids).long().unsqueeze(0))
            batch["all_outside_id"].append(f.all_outside_id)
            if f.lang_id is not None and f.domain_id is not None:
                batch["lang_id"].append(f.lang_id)
                batch["domain_id"].append(f.domain_id)
        
        for k in ("input_ids", "input_mask", "label_ids", "type_ids"):
            batch[k] = torch.cat(batch[k])
        
        for k in ("all_outside_id", "lang_id", "domain_id"):
            if not batch[k]: # because of `None`
                del batch[k]
                continue
            batch[k] = torch.tensor(batch[k]).long()
        
        return batch
    
    @staticmethod
    def clm_collate_fn(data):
        batch = dict(input_ids=list(), input_mask=list())
        for item in data:
            if isinstance(item, tuple) or isinstance(item, list):
                f = item[0]
            else:
                f = item
            batch["input_ids"].append(torch.tensor(f.input_ids).long().unsqueeze(0))
            batch["input_mask"].append(torch.tensor(f.input_mask).long().unsqueeze(0))
        
        for k in ("input_ids", "input_mask"):
            batch[k] = torch.cat(batch[k])
        
        return batch


class ForeverDataIterator:
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
    
    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data
    
    @property
    def dataset(self):
        return self.data_loader.dataset
    
    def __len__(self):
        return len(self.data_loader)


class BaseDataLoader:
    
    @staticmethod
    def forever(data_loader):
        return ForeverDataIterator(data_loader)
    
    @staticmethod
    def trim_dataset_(dataset, max_examples=-1):
        # trim datasets to provided max size (by default consider 
        # full dataset), note the operation is inplace
        if 0. < max_examples < 1.:
            max_examples = int(max_examples * len(dataset))
        if max_examples > 0 and max_examples < len(dataset):
            random.shuffle(dataset.features)
            dataset.features = dataset.features[:max_examples]
    
    @staticmethod
    def split_dataset(dataset, split_ratio=0.5):
        num_split_examples = int(split_ratio * len(dataset))
        random.shuffle(dataset.features)
        split_a = dataset.features[:num_split_examples]
        split_b = dataset.features[num_split_examples:]
        return split_a, split_b


class NERDataLoader(BaseDataLoader):
    
    @staticmethod
    def train_dataloader(
        dataset,
        batch_size=32,
        drop_last=False,
        max_examples=-1,
        forever=False,
        shuffle=False
    ):
        BaseDataLoader.trim_dataset_(dataset, max_examples=max_examples)
        if shuffle:
            random.shuffle(dataset.features)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=RandomSampler(dataset), 
            collate_fn=NERDataset.collate_fn, 
            drop_last=drop_last
        )
        if forever:
            dataloader = BaseDataLoader.forever(dataloader)
        return dataloader
    
    @staticmethod
    def dev_dataloader(dataset, batch_size=32):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=SequentialSampler(dataset), 
            collate_fn=NERDataset.collate_fn
        )
        return dataloader
    
    @staticmethod
    def ssl_train_dataloaders(
        dataset=None,
        sup_dataset=None,
        unsup_dataset=None,
        split_ratio=0.5,
        batch_size=32,
        drop_last=False,
        max_examples=-1,
        forever=False,
        shuffle=False,
        k=1
    ):
        if dataset is not None:
            sup_feats, unsup_feats = BaseDataLoader.split_dataset(dataset, split_ratio)
            sup_dataset = NERDataset(
                sup_feats, 
                id2label=dataset.id2label, id2type=dataset.id2type, 
                unique_name=dataset.unique_name
            )
            unsup_dataset = NERDataset(
                unsup_feats, 
                id2label=dataset.id2label, id2type=dataset.id2type, 
                unique_name=dataset.unique_name
            )
        else:
            assert sup_dataset is not None and unsup_dataset is not None
        outputs = dict(sup=list(), unsup=list())
        # We put the request for multiple dataloaders in ssl setting here to avoid
        # having supervised and unsupervised splits mixup upon multiple calls
        for _ in range(k):
            for split_type, dataset in (("sup", sup_dataset), ("unsup", unsup_dataset)):
                if shuffle:
                    random.shuffle(dataset.features)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler=RandomSampler(dataset), 
                    collate_fn=NERDataset.collate_fn, 
                    drop_last=drop_last
                )
                if forever:
                    dataloader = BaseDataLoader.forever(dataloader)
                outputs[split_type].append(dataloader)
        return outputs
    
    @staticmethod
    def clm_dataloader(dataset, batch_size=32):
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler, 
            collate_fn=NERDataset.clm_collate_fn,
        )
        dataloader = BaseDataLoader.forever(dataloader)
        return dataloader


class MultiNERDataset:
    
    def __init__(self, features):
        self.features = features
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    @staticmethod
    def _collate_fn(data, private=False):
        batch = dict()
        
        for f, private_head_name, shared_head_name in data:
            if private or shared_head_name is None:
                pred_layer = private_head_name
            else:
                pred_layer = shared_head_name
            if pred_layer not in batch:
                batch[pred_layer] = dict(
                    input_ids=list(), 
                    input_mask=list(),
                    label_ids=list(), 
                    type_ids=list(),
                    all_outside_id=list(),
                    shared_label_ids=list(), 
                    lang_id=list(), 
                    domain_id=list()
                )
            d = batch[pred_layer]
            d["input_ids"].append(torch.tensor(f.input_ids).long().unsqueeze(0))
            d["input_mask"].append(torch.tensor(f.input_mask).long().unsqueeze(0))
            d["label_ids"].append(torch.tensor(f.label_ids).long().unsqueeze(0))
            d["type_ids"].append(torch.tensor(f.type_ids).long().unsqueeze(0))
            d["all_outside_id"].append(f.all_outside_id)
            d["shared_label_ids"].append(torch.tensor(f.shared_label_ids).long().unsqueeze(0))
            if f.lang_id is not None and f.domain_id is not None:
                d["lang_id"].append(f.lang_id)
                d["domain_id"].append(f.domain_id)
        
        for pred_layer in batch:
            for k in ("input_ids", "input_mask", "label_ids", "type_ids", "shared_label_ids"):
                batch[pred_layer][k] = torch.cat(batch[pred_layer][k])
            for k in ("all_outside_id", "lang_id", "domain_id"):
                if not batch[pred_layer][k]:
                    del batch[pred_layer][k]
                    continue
                batch[pred_layer][k] = torch.tensor(batch[pred_layer][k]).long()
        
        return batch
    
    collate_fn = functools.partialmethod(_collate_fn, private=False)
    private_collate_fn = functools.partialmethod(_collate_fn, private=True)
    
    @staticmethod
    def shared_as_single_collate_fn(data):
        batch = {
            "shared": dict(
                input_ids=list(), 
                input_mask=list(),
                label_ids=list(), 
                type_ids=list(),
                all_outside_id=list(),
                shared_label_ids=list(), 
                lang_id=list(), 
                domain_id=list()
            )
        }
        
        for f, _, _ in data:
            d = batch["shared"]
            d["input_ids"].append(torch.tensor(f.input_ids).long().unsqueeze(0))
            d["input_mask"].append(torch.tensor(f.input_mask).long().unsqueeze(0))
            d["label_ids"].append(torch.tensor(f.label_ids).long().unsqueeze(0))
            d["type_ids"].append(torch.tensor(f.type_ids).long().unsqueeze(0))
            d["all_outside_id"].append(f.all_outside_id)
            d["shared_label_ids"].append(torch.tensor(f.shared_label_ids).long().unsqueeze(0))
            if f.lang_id is not None and f.domain_id is not None:
                d["lang_id"].append(f.lang_id)
                d["domain_id"].append(f.domain_id)
        
        for k in ("input_ids", "input_mask", "label_ids", "type_ids", "shared_label_ids"):
            batch["shared"][k] = torch.cat(batch["shared"][k])
        
        for k in ("all_outside_id", "lang_id", "domain_id"):
            if not batch["shared"][k]:
                del batch["shared"][k]
                continue
            batch["shared"][k] = torch.tensor(batch["shared"][k]).long()
        
        return batch


class MultiNERDataLoader(BaseDataLoader):
    
    @staticmethod
    def train_dataloader(
        dataset,
        batch_size=32,
        drop_last=False,
        max_examples=-1,
        forever=False,
        private=False
    ):
        BaseDataLoader.trim_dataset_(dataset, max_examples=max_examples)
        sampler = RandomSampler(dataset)
        collate_fn = MultiNERDataset.collate_fn if not private else MultiNERDataset.private_collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler, 
            collate_fn=collate_fn,
            drop_last=drop_last
        )
        if forever:
            dataloader = BaseDataLoader.forever(dataloader)
        return dataloader
    
    @staticmethod
    def dev_dataloader(dataset, batch_size=32, private=False):
        collate_fn = MultiNERDataset.collate_fn if not private else MultiNERDataset.private_collate_fn
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=SequentialSampler(dataset), 
            collate_fn=collate_fn
        )
        return dataloader
    
    @staticmethod
    def shared_as_single_dataloader(dataset, batch_size=32):
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            sampler=SequentialSampler(dataset), 
            collate_fn=MultiNERDataset.shared_as_single_collate_fn
        )
        return dataloader
