# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

from ..base import ArgumentsBase
from ..train import BaseTrainer
from ..data import SimpleData


class NERTrainer(BaseTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
        self.load_data()
    
    def load_data(self):
        # Standardize input into right format as metadata
        metadata = self.parse_input_to_dataset_metadata(self.exp_args.train_dataset)
        # Override the max_examples if max_num_train_examples is provided separately
        if self.exp_args.max_num_train_examples > 0:
            metadata["max_examples"] = self.exp_args.max_num_train_examples
        
        data_kwargs = self.data_kwargs
        data_dir = data_kwargs.pop("preprocessed_data_dir")
        
        self.data = SimpleData(data_dir, metadata, **data_kwargs)
        # In this setting we only use validation set of training dataset
        self.register_model_validation_datasets(metadata["name"])
        self.load_eval_data(train_datasets=[metadata["name"],])
        
        # Config kwargs that will be hooked to model config file
        id2label, label2id = self.data.get_label_maps(metadata["name"])
        id2type, type2id = self.data.get_type_maps(metadata["name"])
        
        self.config_kwargs.update(dict(
            label2id=label2id, id2label=id2label,
            type2id=type2id, id2type=id2type
        ))
        
        # For LDAM we need additional information about class frequencies
        if self.model_args.loss_fct == "ldam" and self.training_args.do_train:
            train_dataset = self.data.get_train_dataset(metadata)
            class_num_list = train_dataset.get_class_counts()
            self.config_kwargs["class_num_list"] = class_num_list
    
    def batch_to_device(self, inputs):
        return {tensor_name: tensor.to(self.device) for tensor_name, tensor in inputs.items()}
    
    def before_train(self):
        self.load_train_data()
        self.model_optim_config = dict(
            model=self.model, 
            optim="adamw", 
            lr_scheduler=self.exp_args.lr_scheduler
        )
        self.configure_optimization()
    
    def run_epoch(self):
        train_dataloader = self.train_data["ner"]
        train_iter = self.data.to_iter(train_dataloader)
        
        model = self.model
        
        for self.step in self.epoch_iterator:
            inputs = next(train_iter)
            inputs = self.batch_to_device(inputs)
            model.train()
            
            outputs = model(inputs)
            loss = outputs["ner"]["loss"]
            
            self.model_backward(loss)
            self.stats["batch_train_loss"].append(self.adjust_loss(loss).item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
                self.clip_grad_norm_(model.parameters())
                self.model_update()
                self.global_step += 1
                self.update_lr()
                self.model_zero_grad()
                
                logs = {}
                logs["loss"] = sum(self.stats["batch_train_loss"]) / self.num_batches_seen
                logs["epoch"] = self.progress
                logs["step"] = self.global_step
                
                self.do_logging_step(model, logs, evaluate=True)
                self.do_save_step(model)
            
            if self.is_max_step:
                self.epoch_iterator.close()
                break


@dataclass
class NERArguments(ArgumentsBase):
    """
    Experiment specific arguments.
    """
    preprocessed_data_dir: str = field(
        metadata={"help": "The data directory created by preprocessing script."}
    )
    train_dataset: str = field(
        metadata={"help": "NER training dataset."}
    )
    eval_datasets: list = field(
        default=None,
        metadata={"help": "Additional datasets to evaluate the trained model in zero-shot setting."}
    )
    max_num_train_examples: int = field(
        default=-1,
        metadata={"help": "Maximum no. of training examples to consider (-1 = consider all)."}
    )
    lr_scheduler: str = field(
        default="linear",
        metadata={"help": "Which learning rate scheduler to use."}
    )
