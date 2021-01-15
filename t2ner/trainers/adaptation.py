# -*- coding: utf-8 -*-

from ..train import BaseTrainer
from ..data import SimpleAdaptationData


class AdaptationTrainer(BaseTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
        self.load_data()
    
    def load_data(self):
        src_metadata = self.parse_input_to_dataset_metadata(self.exp_args.src_dataset)
        tgt_metadata = self.parse_input_to_dataset_metadata(self.exp_args.tgt_dataset)
        
        data_kwargs = self.data_kwargs
        data_dir = data_kwargs.pop("preprocessed_data_dir")
        
        self.data = SimpleAdaptationData(data_dir, src_metadata, tgt_metadata, **data_kwargs)
        # In this setting we only use src validation set of training dataset
        self.register_model_validation_datasets(src_metadata["name"])
        self.load_eval_data(train_datasets=[src_metadata["name"], tgt_metadata["name"]])
        
        # Config kwargs that will be hooked to model config file
        id2label, label2id = self.data.get_label_maps(src_metadata["name"])
        id2type, type2id = self.data.get_type_maps(src_metadata["name"])
        
        update_config_kwargs = dict(
            label2id=label2id, id2label=id2label,
            type2id=type2id, id2type=id2type
        )
        if hasattr(self.exp_args, "pooling"):
            update_config_kwargs["pooling"] = self.exp_args.pooling
        self.config_kwargs.update(update_config_kwargs)
    
    def batch_to_device(self, inputs):
        return {tensor_name: tensor.to(self.device) for tensor_name, tensor in inputs.items()}
    
    def _get_batch(self, src_or_tgt, task="ner", dataloader_idx=0):
        if task == "ner":
            name = "{}_ner_{}_train_iter".format(src_or_tgt, dataloader_idx)
            if not hasattr(self, name):
                dataloader = self.train_data[task][src_or_tgt][dataloader_idx]
                diter = self.data.to_iter(dataloader)
                self.__setattr__(name, diter)
            else:
                diter = self.__getattribute__(name)
        else:
            name = "{}_{}_train_iter".format(src_or_tgt, task)
            if not hasattr(self, name):
                dataloader = self.train_data[task][src_or_tgt]
                diter = self.data.to_iter(dataloader)
                self.__setattr__(name, diter)
            else:
                diter = self.__getattribute__(name)
        
        return self.batch_to_device(next(diter))
    
    def get_src_batch(self, task="ner", dataloader_idx=0):
        return self._get_batch("src", task, dataloader_idx)
    
    def get_tgt_batch(self, task="ner", dataloader_idx=0):
        return self._get_batch("tgt", task, dataloader_idx)
