# -*- coding: utf-8 -*-

import os
import collections
import random

from . import utils
from . import ner
from .base import BaseData


class MultiData(BaseData):
    
    def __init__(self, preprocessed_data_dir, train_datasets_metadata_list, **kwargs):
        super().__init__(preprocessed_data_dir, **kwargs)
        self.metadata = self.collect_metadata(train_datasets_metadata_list)
    
    def collect_metadata(self, train_datasets_metadata_list):
        unique_names = sorted(list(set([metadata["name"] for metadata in train_datasets_metadata_list])))
        
        langs, domains, sources = zip(*[dataset.split(".") for dataset in unique_names])
        langs = sorted(list(set(langs)))
        num_langs = len(langs)
        
        domains = sorted(list(set(domains)))
        num_domains = len(domains)
        
        sources = sorted(list(set(sources)))
        num_sources = len(sources)
        
        # Identify mode (S: single, M: multi, L: lang, D: domain)
        if num_langs == 1 and num_domains == 1:
            mode = "SLSD"
        elif num_langs == 1 and num_domains > 1:
            mode = "SLMD"
        elif num_langs > 1 and num_domains == 1:
            mode = "MLSD"
        elif num_langs > 1 and num_domains > 1:
            mode = "MLMD"
        else:
            raise ValueError("Invalid mode for multi-dataset.")
        
        dataset_info = dict()
        all_labels = set()
        for metadata in train_datasets_metadata_list:
            unique_name = metadata["name"]
            # possible duplication
            if unique_name in dataset_info:
                continue
            label_list = self.get_label_list(unique_name)
            type_list = self.get_type_list(unique_name)
            lang, domain, source = unique_name.split(".")
            dataset_info[unique_name] = {
                "name": unique_name.replace(".", "_"),
                "domain": domain,
                "lang": lang,
                "source": source,
                "label_list": label_list,
                "num_labels": len(label_list),
                "type_list": type_list,
                "num_types": len(type_list),
                "metadata": metadata
            }
            all_labels.update(label_list)
        
        metadata = {
            "mode": mode,
            "langs": langs,
            "domains": domains,
            "sources": sources,
            "dataset_info": dataset_info,
            "all_labels": sorted(list(all_labels))
        }
        
        return metadata
    
    @property
    def type(self):
        return self.metadata["mode"]
    
    @property
    def langs(self):
        return self.metadata["langs"]
    
    @property
    def num_langs(self):
        return len(self.langs)
    
    @property
    def lang2id(self):
        return {lang: idx for idx, lang in enumerate(self.langs)}
    
    @property
    def id2lang(self):
        return {idx: lang for lang, idx in self.lang2id.items()}
    
    @property
    def domains(self):
        return self.metadata["domains"]
    
    @property
    def num_domains(self):
        return len(self.domains)
    
    @property
    def domain2id(self):
        return {domain: idx for idx, domain in enumerate(self.domains)}
    
    @property
    def id2domain(self):
        return {idx: domain for domain, idx in self.domain2id.items()}
    
    @property
    def sources(self):
        return self.metadata["sources"]
    
    @property
    def num_sources(self):
        return len(self.sources)
    
    @property
    def shared_labels(self):
        return self.metadata["all_labels"]
    
    @property
    def total_shared_labels(self):
        return len(self.shared_labels)
    
    @property
    def shared_label2id(self):
        return {label: idx for idx, label in enumerate(sorted(self.shared_labels))}
    
    @property
    def shared_id2label(self):
        return {idx: label for label, idx in self.shared_label2id.items()}
    
    def __getitem__(self, unique_name):
        return self.metadata["dataset_info"][unique_name]
    
    def __iter__(self):
        for unique_name, dataset_info in self.metadata["dataset_info"].items():
            yield unique_name, dataset_info
    
    @property
    def labels_str_to_datasets(self):
        labels_str_to_datasets = collections.defaultdict(list)
        
        for unique_name, dataset_info in self:
            label_list = dataset_info["label_list"]
            labels_str = "\t".join(label_list)
            labels_str_to_datasets[labels_str].append(unique_name)
        
        return labels_str_to_datasets
    
    @property
    def dataset_to_labels_str(self):
        return {
            unique_name: labels_str
            for labels_str, unique_names in self.labels_str_to_datasets.items()
            for unique_name in unique_names
        }
    
    def register_training_datasets_with_labels(self, dataset_names):
        dataset_names = set(dataset_names)
        train_datasets_names_with_labels = list()
        
        for unique_name, _ in self:
            if unique_name in dataset_names:
                train_datasets_names_with_labels.append(unique_name)
        
        if not train_datasets_names_with_labels:
            raise ValueError(
                "Could not find any training dataset in the original collection."
            )
        
        self.train_datasets_names_with_labels = train_datasets_names_with_labels
    
    def register_training_datasets_without_labels(self, dataset_names):
        dataset_names = set(dataset_names)
        train_datasets_names_without_labels = list()
        
        for unique_name, _ in self:
            if unique_name in dataset_names:
                train_datasets_names_without_labels.append(unique_name)
        
        if not train_datasets_names_without_labels:
            raise ValueError(
                "Could not find any training dataset in the original collection."
            )
        
        self.train_datasets_names_without_labels = train_datasets_names_without_labels
    
    def _load_datasets(self, dataset_names, split):
        unique_name2train_dataset = dict()
        for unique_name in dataset_names:
            datasets = self.get_datasets(unique_name=unique_name, splits=(split,))
            dataset = list(datasets)[0][1]
            max_examples = self[unique_name]["metadata"].get("max_examples", -1)
            ner.BaseDataLoader.trim_dataset_(dataset, max_examples=max_examples)
            unique_name2train_dataset[unique_name] = dataset
        return unique_name2train_dataset
    
    @property
    def train_datasets_labeled(self):
        return self._load_datasets(self.train_datasets_names_with_labels, "train")
    
    @property
    def train_datasets_unlabeled(self):
        return self._load_datasets(self.train_datasets_names_without_labels, "train")
    
    @property
    def heads_info(self):
        dataset_to_head_info = dict()
        dataset_to_labels_str = self.dataset_to_labels_str
        labels_strs = sorted(list({labels_str for labels_str in dataset_to_labels_str.values()}))
        labels_str2id = {labels_str: idx for idx, labels_str in enumerate(labels_strs)}
        
        for unique_name, dataset_info in self:
            dataset_to_head_info[unique_name] = {
                "private_head_name": dataset_info["name"],
                "shared_head_name": str(labels_str2id[dataset_to_labels_str[unique_name]]),
                "num_labels": dataset_info["num_labels"],
                "num_types": dataset_info["num_types"]
            }
        
        return dataset_to_head_info
    
    def convert_to_shared_label_ids(self, shared_label2id, label_ids, ner_dataset):
        # Transform `label_ids` of a given dataset to `shared_label_ids`
        labels = ner_dataset.decode_label_ids(label_ids)
        # l == int implies pad id (e.g. -100), otherwise label string
        shared_label_ids = [shared_label2id[l] if isinstance(l, str) else l for l in labels]
        return shared_label_ids
    
    def _unioned_datasets(self, unique_name2train_dataset):
        features = list()
        private_head_names = list()
        shared_head_names = list()
        heads_info = self.heads_info
        shared_label2id = self.shared_label2id
        
        for unique_name in unique_name2train_dataset:
            dataset_info = self[unique_name]
            dataset = unique_name2train_dataset[unique_name]
            
            # Add shared label ids
            for f in dataset.features:
                f.shared_label_ids = self.convert_to_shared_label_ids(
                    shared_label2id, f.label_ids, dataset
                )
            
            lang_id = self.lang2id[dataset_info["lang"]]
            domain_id = self.domain2id[dataset_info["domain"]]
            dataset.add_metadata(lang_id, domain_id)
            features.extend(dataset.features)
            private_head_names.extend([heads_info[unique_name]["private_head_name"]] * len(dataset))
            shared_head_names.extend([heads_info[unique_name]["shared_head_name"]] * len(dataset))
        
        # We need to pass this information here to make ``collate_fn`` independent
        features = list(zip(features, private_head_names, shared_head_names))
        random.shuffle(features)
        return ner.MultiNERDataset(features)
    
    @property
    def train_metadata(self):
        return self._train_metadata
    
    @train_metadata.setter
    def train_metadata(self, metadata):
        self._train_metadata = metadata
    
    def get_train_dataloader(self, metadata):
        mix_method = metadata.get("mix_method", "union")
        if mix_method == "union":
            dataset = self._unioned_datasets(self.train_datasets_labeled)
        else:
            raise NotImplementedError
        
        dataloader = ner.MultiNERDataLoader.train_dataloader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=metadata.get("drop_last", False),
            max_examples=metadata.get("max_examples", -1),
            forever=metadata.get("forever", False),
            private=metadata.get("private", False)
        )
        
        return dataloader
    
    def get_train_data(self, lm=False, lm_dataset=None):
        train_dataloaders = dict()
        metadata = self.train_metadata
        
        train_dataloaders["ner"] = self.get_train_dataloader(metadata)
        train_dataloader_len = len(train_dataloaders["ner"])
        num_train_examples = len(train_dataloaders["ner"].dataset)
        
        if lm:
            train_dataloaders["lm"] = self.get_clm_train_dataloader(
                lm_dataset=train_dataloaders["ner"].dataset if lm_dataset is None else lm_dataset
            )
        
        return train_dataloaders, train_dataloader_len, num_train_examples
    
    def get_eval_and_test_datasets(self, eval_datasets):
        split2dataset = dict(dev=dict(), test=dict())
        shared_label2id = self.shared_label2id
        
        for eval_unique_name in set(eval_datasets):
            splits = ("dev", "test")
            split2dataset_i = {split: dataset 
                for split, dataset in self.get_datasets(eval_unique_name, splits)
            }
            for split in splits:
                if split not in split2dataset_i:
                    continue
                dataset = split2dataset_i[split]
                # Add shared label ids
                for f in dataset.features:
                    f.shared_label_ids = self.convert_to_shared_label_ids(
                        shared_label2id, f.label_ids, dataset
                    )
                
                split2dataset[split][eval_unique_name] = dataset
        
        return split2dataset
    
    def get_eval_and_test_dataloaders(self, eval_datasets, **kwargs):
        do_eval = kwargs.get("do_eval", True)
        do_predict = kwargs.get("do_predict", False)
        all_shared = kwargs.get("all_shared", False)
        private = kwargs.get("private", False)
        
        split2dataset = self.get_eval_and_test_datasets(eval_datasets)
        eval_data = dict()
        test_data = dict()
        heads_info = self.heads_info
        labels_str_to_datasets = self.labels_str_to_datasets
        
        for split in split2dataset:
            for eval_unique_name in split2dataset[split]:
                dataset = split2dataset[split][eval_unique_name]
                add_dataset_for_shared_only = False
                
                # Find the prediction layer for eval set and zip with the features
                label2id = {label: idx for idx, label in dataset.id2label.items()}
                try:
                    labels_str = "\t".join(sorted(list(label2id.keys())))
                    same_labels_datasets = labels_str_to_datasets[labels_str]
                except KeyError:
                    raise RuntimeError(
                        "{}-{} dataset does not have label set seen in training."
                        .format(split, eval_unique_name)
                    )
                else:
                    if not private:
                        # When same prediction layer is used in multi-dataset
                        # setting with same labels, we can take any dataset
                        # name as they have same layer so the index [0]
                        head_info = heads_info[same_labels_datasets[0]]
                    else:
                        zero_shot = eval_unique_name not in heads_info
                        if zero_shot:
                            # When we use different layers for each dataset, despite
                            # having same label set and we come across a zero-shot dataset 
                            head_info = None
                        else:
                            head_info = heads_info[eval_unique_name]
                
                if head_info is None:
                    private_head_name = "unk"
                    shared_head_name = None
                else:
                    private_head_name = head_info["private_head_name"]
                    shared_head_name = head_info["shared_head_name"]
                
                dataset.features = list(zip(
                    dataset.features, 
                    [private_head_name] * len(dataset), 
                    [shared_head_name] * len(dataset)
                ))
                
                if split == "dev" and do_eval:
                    eval_data[eval_unique_name] = (
                        ner.MultiNERDataLoader
                        .dev_dataloader(dataset, self.eval_batch_size, private=private),
                        dataset.id2label
                    )
                    
                    if all_shared:
                        eval_data[eval_unique_name + ".shared"] = (
                            ner.MultiNERDataLoader
                            .shared_as_single_dataloader(dataset),
                            self.shared_id2label
                        )
                
                elif split == "test" and do_predict:
                    test_data[eval_unique_name] = (
                        ner.MultiNERDataLoader
                        .dev_dataloader(dataset, self.eval_batch_size, private=private),
                        dataset.id2label
                    )
                    
                    if all_shared:
                        test_data[eval_unique_name + ".shared"] = (
                            ner.MultiNERDataLoader
                            .shared_as_single_dataloader(dataset),
                            self.shared_id2label
                        )
        
        return eval_data, test_data
