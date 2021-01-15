# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import GRL, MaskedSoftmax
from ..adaptation import AdaptationTrainer


class MMETrainer(AdaptationTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data()
        # Note the model is composed of <trasnformers encoder, (xnet), classifier>
        model = self.model
        
        try:
            assert not model.config.use_crf
            assert model.config.loss_fct == "ce"
        except:
            raise NotImplementedError
        
        F = nn.ModuleList([model.model]) # encoder
        if hasattr(model, "xnet"):
            F.append(model.xnet)
        
        C = model.classifier
        self.register_model(C, name="C")
        self.revgrad = GRL(self.exp_args.grl_coeff)
        
        model_optim_config = dict()
        extended_optim_config_dict = dict()
        
        for name, lr, sched_name in [
            ("model", self.exp_args.learning_rate_F, self.exp_args.lr_scheduler_F),
            ("C", self.exp_args.learning_rate_C, self.exp_args.lr_scheduler_C)
        ]:
            optim_config = dict(
                optim="adamw", optim_kwargs={"lr": lr}, lr_scheduler=sched_name
            )
            if name == "model":
                optim_config["model"] = F
                model_optim_config = optim_config
            
            elif name == "C":
                optim_config["C"] = C
                extended_optim_config_dict[name] = optim_config
        
        self.model_optim_config = model_optim_config
        self.extended_optim_config_dict = extended_optim_config_dict
        self.configure_optimization()
    
    def run_epoch(self):
        model = self._models["model"]
        C = self._models["C"]
        
        metrics = ["batch_loss_tgt"]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        for self.step in self.epoch_iterator:
            model.train()
            C.train()
            
            src, tgt = self.get_src_batch(), self.get_tgt_batch()
            
            # Source NER inputs
            model_outputs = model(src)
            clf_loss = model_outputs["ner"]["loss"]
            
            self.model_backward(clf_loss)
            
            if self.is_update_step:
                self.model_update()
                self.model_zero_grad()
            
            tgt_seq = model(tgt, tag=False)["encoder"]["seq"]
            tgt_seq = self.revgrad(tgt_seq)
            tgt_logits = C(tgt_seq)["logits"]
            tgt_prob = MaskedSoftmax()(tgt_logits, mask=tgt.get("input_mask", None))
            loss_tgt = -(-tgt_prob * torch.log(tgt_prob + 1e-5)).sum(2).sum(1).mean()
            
            self.model_backward(loss_tgt * self.exp_args.lmbda)
            self.stats["batch_train_loss"].append(self.adjust_loss(clf_loss).item())
            self.stats["batch_loss_tgt"].append(self.adjust_loss(loss_tgt).item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
                self.model_update()
                self.global_step += 1
                self.update_lr()
                self.model_zero_grad()
                
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
class MMEArguments(ArgumentsBase):
    """
    Experiment specific arguments.
    """
    preprocessed_data_dir: str = field(
        metadata={"help": "The data directory created by preprocessing script."}
    )
    src_dataset: str = field(
        metadata={"help": "NER source training dataset."}
    )
    tgt_dataset: str = field(
        metadata={"help": "NER target training dataset."}
    )
    eval_datasets: list = field(
        default=None,
        metadata={"help": "Additional datasets to evaluate the trained model in zero-shot setting."}
    )
    learning_rate_F: float = field(
        default=6e-6,
        metadata={"help": "Task learnign rate."}
    )
    lr_scheduler_F: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use."}
    )
    learning_rate_C: float = field(
        default=5e-4,
        metadata={"help": "Discriminator learning rate."}
    )
    lr_scheduler_C: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for discriminator."}
    )
    grl_coeff: float = field(
        default=1.0,
        metadata={"help": "Fixed coefficient for grl training (-1 will use scheduler)."}
    )
    lmbda: float = field(
        default=1.0,
        metadata={"help": "Trade-off parameter between task and adversarial loss."}
    )
