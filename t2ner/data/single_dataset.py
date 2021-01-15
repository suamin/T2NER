# -*- coding: utf-8 -*-

from . import ner
from .base import BaseData


class SimpleData(BaseData):
    
    def __init__(self, preprocessed_data_dir, train_dataset_metadata, **kwargs):
        super().__init__(preprocessed_data_dir, **kwargs)
        self.metadata = self.collect_metadata(train_dataset_metadata)
    
    def collect_metadata(self, train_dataset_metadata):
        unique_name = train_dataset_metadata["name"]
        label_list = self.get_label_list(unique_name)
        type_list = self.get_type_list(unique_name)
        lang, domain, source = unique_name.split(".")
        
        metadata = {
            "name": train_dataset_metadata["name"].replace(".", "_"),
            "lang": lang,
            "domain": domain,
            "source": source,
            "label_list": label_list,
            "num_labels": len(label_list),
            "type_list": type_list,
            "num_types": len(type_list),
            "metadata": train_dataset_metadata
        }
        
        return metadata
    
    @property
    def lang(self):
        return self.metadata["lang"]
    
    @property
    def domain(self):
        return self.metadata["domain"]
    
    def get_train_data(self, lm=False, lm_dataset=None):
        train_dataloaders = dict()
        metadata = self.metadata["metadata"]
        
        train_dataloaders["ner"] = self.get_train_dataloader(metadata)
        train_dataloader_len = len(train_dataloaders["ner"])
        num_train_examples = len(train_dataloaders["ner"].dataset)
        
        if lm:
            train_dataloaders["lm"] = self.get_clm_train_dataloader(metadata, lm_dataset)
        
        return train_dataloaders, train_dataloader_len, num_train_examples
