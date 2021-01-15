# -*- coding: utf-8 -*-

import torch

from ..train import BaseTrainer
from ..data import SemiSupervisedData


class SSLTrainer(BaseTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
        self.load_data()
    
    def load_data(self):
        metadata = self.parse_input_to_dataset_metadata(self.exp_args.train_dataset)
        
        data_kwargs = self.data_kwargs
        data_dir = data_kwargs.pop("preprocessed_data_dir")
        
        self.data = SemiSupervisedData(data_dir, metadata, **data_kwargs)
        self.register_model_validation_datasets(metadata["name"])
        self.load_eval_data(train_datasets=[metadata["name"],])
        
        id2label, label2id = self.data.get_label_maps(metadata["name"])
        id2type, type2id = self.data.get_type_maps(metadata["name"])
        
        self.config_kwargs.update(dict(
            label2id=label2id, id2label=id2label,
            type2id=type2id, id2type=id2type
        ))
    
    def batch_to_device(self, inputs):
        return {tensor_name: tensor.to(self.device) for tensor_name, tensor in inputs.items()}
    
    def _get_batch(self, sup_or_unsup, task="ner", dataloader_idx=0):
        if task == "ner":
            name = "{}_ner_{}_train_iter".format(sup_or_unsup, dataloader_idx)
            if not hasattr(self, name):
                dataloader = self.train_data[task][sup_or_unsup][dataloader_idx]
                diter = self.data.to_iter(dataloader)
                self.__setattr__(name, diter)
            else:
                diter = self.__getattribute__(name)
        else:
            name = "{}_train_iter".format(task)
            if not hasattr(self, name):
                dataloader = self.train_data[task]
                diter = self.data.to_iter(dataloader)
                self.__setattr__(name, diter)
            else:
                diter = self.__getattribute__(name)
        
        return self.batch_to_device(next(diter))
    
    def get_sup_batch(self, task="ner", dataloader_idx=0):
        return self._get_batch("sup", task, dataloader_idx)
    
    def get_unsup_batch(self, task="ner", dataloader_idx=0):
        return self._get_batch("unsup", task, dataloader_idx)
