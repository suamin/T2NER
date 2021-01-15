# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import MaskedSoftmax
from ..semisupervised import SSLTrainer


class EntMinTrainer(SSLTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data(split_ratio=self.exp_args.split_ratio)
        self.model_optim_config = dict(
            model=self.model, 
            optim="adamw", 
            lr_scheduler=self.exp_args.lr_scheduler
        )
        self.configure_optimization()
    
    def run_epoch(self):
        metrics = ["batch_train_loss", "batch_sup_loss", "batch_unsup_loss"]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        model = self.model
        
        for self.step in self.epoch_iterator:
            model.train()
            
            # Batch of supervised and unsupervised inputs
            sup, unsup = self.get_sup_batch(), self.get_unsup_batch()
            
            sup_outputs = model(sup)
            sup_loss = sup_outputs["ner"]["loss"]
            
            # Remove labels so classifier don't expect labels
            unsup["label_ids"] = None
            
            unsup_outputs = MaskedSoftmax()(
                logits=model(unsup)["ner"]["logits"],
                mask=unsup.get("input_mask", None),
                dim=2
            ) # batch_size x seq_len x num_classes
            unsup_loss = (-unsup_outputs * torch.log(unsup_outputs + 1e-5)).sum(2).sum(1).mean()
            
            loss = self.joint_loss(sup_loss, aux_losses=unsup_loss, lmbda=self.exp_args.lmbda)
            
            self.model_backward(loss)
            
            self.stats["batch_train_loss"].append(loss.item())
            self.stats["batch_sup_loss"].append(self.adjust_loss(sup_loss).item())
            self.stats["batch_unsup_loss"].append(self.adjust_loss(unsup_loss).item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
                self.clip_grad_norm_(model.parameters())
                self.model_update(["model"])
                self.global_step += 1
                self.update_lr(["model"])
                self.model_zero_grad(["model"])
                
                logs = {}
                logs["loss"] = sum(self.stats["batch_train_loss"]) / self.num_batches_seen
                for k in metrics:
                    logs[k[6:]] = sum(self.stats[k]) / self.num_batches_seen
                logs["epoch"] = self.progress
                logs["step"] = self.global_step
                
                self.do_logging_step(model, logs, evaluate=True)
                self.do_save_step(model)
            
            if self.is_max_step:
                self.epoch_iterator.close()
                break


@dataclass
class EntMinArguments(ArgumentsBase):
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
    lr_scheduler: str = field(
        default="linear",
        metadata={"help": "Which learning rate scheduler to use."}
    )
    split_ratio: float = field(
        default=0.1,
        metadata={"help": "Supervised / unsupervised split ratio."}
    )
    lmbda: float = field(
        default=1e-3,
        metadata={"help": "Trade-off parameter between supervised and unsupervised loss."}
    )
