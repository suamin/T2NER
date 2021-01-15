# -*- coding: utf-8 -*-

import torch.nn as nn

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import BinaryAdvClassifier
from ..adaptation import AdaptationTrainer


class GRLTrainer(AdaptationTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data()
        model = self.model
        # Create GRL discriminator
        disc = BinaryAdvClassifier(
            model.config.hidden_size,
            coeff=self.exp_args.grl_coeff,
            **{
                "gamma": self.exp_args.grl_scheduler_gamma,
                "max_iter": self.exp_args.grl_scheduler_max_iter,
                "lo": self.exp_args.grl_scheduler_lo,
                "hi": self.exp_args.grl_scheduler_hi
            }
        )
        self.register_model(disc, name="disc")
        # Joint optimization setup
        self.model_optim_config = dict(
            model=nn.ModuleList([model, disc]), 
            optim="adamw", 
            lr_scheduler=self.exp_args.lr_scheduler
        )
        self.configure_optimization()
    
    def run_epoch(self):
        model = self._models["model"]
        disc = self._models["disc"]
        
        metrics = ["batch_clf_loss", "batch_disc_loss", "batch_disc_acc"]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        for self.step in self.epoch_iterator:
            model.train()
            disc.train()
            
            src, tgt = self.get_src_batch(), self.get_tgt_batch()
            
            # Source NER inputs
            model_outputs = model(src)
            clf_loss = model_outputs["ner"]["loss"]
            
            # Adversarial inputs
            xs = model_outputs["encoder"]["pooled"]
            # Turn off tagging for tgt
            xt = model(tgt, tag=False)["encoder"]["pooled"]
            disc_outputs = disc(xs, xt)
            disc_loss = disc_outputs["loss"]
            if self.n_gpu > 1:
                disc_outputs["acc"] = disc_outputs["acc"].mean()
            
            loss = clf_loss + self.exp_args.lmbda * disc_loss
            self.model_backward(loss)
            
            self.stats["batch_train_loss"].append(self.adjust_loss(loss).item())
            self.stats["batch_clf_loss"].append(self.adjust_loss(clf_loss).item())
            self.stats["batch_disc_loss"].append(self.adjust_loss(disc_loss).item())
            self.stats["batch_disc_acc"].append(disc_outputs["acc"].item())
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
class GRLArguments(ArgumentsBase):
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
    pooling: str = field(
        default="mean",
        metadata={"help": "Sequence pooling method."}
    )
    lr_scheduler: str = field(
        default="linear",
        metadata={"help": "Which learning rate scheduler to use."}
    )
    grl_coeff: float = field(
        default=1e-5,
        metadata={"help": "Fixed coefficient for grl training (-1 will use scheduler)."}
    )
    grl_scheduler_gamma: float = field(
        default=10.0,
        metadata={"help": "The value of gamma parameter in grl scheduler."}
    )
    grl_scheduler_max_iter: float = field(
        default=1000.,
        metadata={"help": "No. of iterations grl need to reach from start to end."}
    )
    grl_scheduler_lo: float = field(
        default=0.,
        metadata={"help": "Start value of grl coeff."}
    )
    grl_scheduler_hi: float = field(
        default=1.,
        metadata={"help": "End value of grl coeff."}
    )
    lmbda: float = field(
        default=1.0,
        metadata={"help": "Trade-off parameter between task and adversarial loss."}
    )
