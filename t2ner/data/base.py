# -*- coding: utf-8 -*-

import os

from abc import ABC, abstractmethod

from . import ner
from . import utils


class BaseData(ABC):
    
    def __init__(self, preprocessed_data_dir, **kwargs):
        self.preprocessed_data_dir = preprocessed_data_dir
        self.train_batch_size = kwargs.get("train_batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 32)
        self.features_cache_dir = kwargs.get("features_cache_dir", None)
        self.overwrite_cache = kwargs.get("overwrite_cache", False)
        self.features_kwargs = kwargs.get("features_kwargs", dict())
    
    def get_datasets(self, unique_name, splits):
        # Returns a sequence of (split, dataset) tuples
        datasets = ner.NERDataset.load_and_cache_features_as_datasets(
            preprocessed_data_dir=self.preprocessed_data_dir,
            unique_name=unique_name,
            splits=splits,
            features_cache_dir=self.features_cache_dir,
            overwrite_cache=self.overwrite_cache,
            **self.features_kwargs
        )
        return datasets
    
    def get_label_list(self, unique_name):
        label_file = os.path.join(self.preprocessed_data_dir, "{}.labels".format(unique_name))
        label_list = utils.read_labels(label_file)
        label_list = sorted(list(set(label_list)))
        return label_list
    
    def get_label_maps(self, unique_name):
        label_list = self.get_label_list(unique_name)    
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for label, idx in label2id.items()}
        return id2label, label2id
    
    def get_type_list(self, unique_name):
        label_list = self.get_label_list(unique_name)
        type_list = [label[2:] for label in label_list if label != "O"] + ["O"]
        type_list = sorted(list(set(type_list)))
        return type_list
    
    def get_type_maps(self, unique_name):
        type_list = self.get_type_list(unique_name)
        type2id = {etype: idx for idx, etype in enumerate(type_list)}
        id2type = {idx: etype for etype, idx in type2id.items()}
        return id2type, type2id
    
    @abstractmethod
    def get_train_data(self, *args, **kwargs):
        pass
    
    def get_train_dataset(self, metadata):
        # Returns an iterator of (split, dataset)
        datasets = self.get_datasets(unique_name=metadata["name"], splits=("train",))
        dataset = list(datasets)[0][1]
        return dataset
    
    def get_dataset(self, metadata=None, dataset=None):
        if metadata is not None:
            dataset = self.get_train_dataset(metadata)
        elif dataset is not None:
            pass
        else:
            raise ValueError(
                "At least metadata or dataset should be provided"
            )
        return dataset
    
    def _get_train_dataloader(self, dataset, metadata):
        dataloader = ner.NERDataLoader.train_dataloader(
            dataset=dataset,
            batch_size=self.train_batch_size,
            drop_last=metadata.get("drop_last", False),
            max_examples=metadata.get("max_examples", -1),
            forever=metadata.get("forever", True),
            shuffle=metadata.get("shuffle", False)
        )
        return dataloader
    
    def get_train_dataloader(self, metadata=None, dataset=None):
        if metadata is None:
            metadata = dict()
        dataset = self.get_dataset(metadata, dataset)
        return self._get_train_dataloader(dataset, metadata)
    
    def get_k_train_dataloaders(self, metadata, k=1):
        dataloaders = list()
        dataset = self.get_train_dataset(metadata)
        for i in range(k):
            if i == 0:
                metadata["shuffle"] = False
            else:
                metadata["shuffle"] = True
            dataloader = self.get_train_dataloader(metadata)
            dataloaders.append(dataloader)
        return dataloaders
    
    def _get_ssl_train_dataloaders(self, dataset, split_ratio, k):
        ssl_dataloaders = ner.NERDataLoader.ssl_train_dataloaders(
            dataset=dataset,
            split_ratio=split_ratio,
            batch_size=self.train_batch_size,
            drop_last=True,
            max_examples=-1,
            forever=True,
            shuffle=True,
            k=k
        )
        return ssl_dataloaders
    
    def get_ssl_train_dataloaders(self, metadata=None, dataset=None, k=1, split_ratio=0.5):
        dataset = self.get_dataset(metadata, dataset)
        dataloaders = self._get_ssl_train_dataloaders(dataset, split_ratio, k)
        return dataloaders
    
    @staticmethod
    def to_iter(dataloader):
        try:
            iter_dataloader = iter(dataloader)
        except:
            # if forever option was used, it is already an iterator
            iter_dataloader = dataloader
        return iter_dataloader
    
    def get_clm_train_dataloader(self, metadata=None, dataset=None):
        assert metadata is not None or dataset is not None
        if dataset is None:
            dataset = self.get_train_dataset(metadata)
        dataloader = ner.NERDataLoader.lm_dataloader(dataset, batch_size=self.train_batch_size)
        return dataloader
    
    def get_eval_and_test_dataloaders(self, datasets, **kwargs):
        splits = ()
        eval_data = dict()
        do_eval = kwargs.get("do_eval", True)
        if do_eval:
            splits += ("dev",)
        
        test_data = dict()
        do_predict = kwargs.get("do_predict", False)
        if do_predict:
            splits += ("test",)
        
        for unique_name in set(datasets):
            split2dataset = dict()
            if splits:
                for split, dataset in self.get_datasets(unique_name, splits):
                    split2dataset[split] = dataset
            
            for split in splits:
                # Datasets can only have dev or test splits
                if split not in split2dataset:
                    continue
                dataset = split2dataset[split]
                
                if split == "dev":
                    dataloader = ner.NERDataLoader.dev_dataloader(dataset, self.eval_batch_size)
                    eval_data[unique_name] = (dataloader, dataset.id2label)
                
                elif split == "test":
                    dataloader = ner.NERDataLoader.dev_dataloader(dataset, self.eval_batch_size)
                    test_data[unique_name] = (dataloader, dataset.id2label)
        
        return eval_data, test_data
