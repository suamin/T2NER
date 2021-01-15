# -*- coding: utf-8 -*-

import torch.nn as nn

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import BinaryAdvClassifier
from ..adaptation import AdaptationTrainer


class KeungTrainer(AdaptationTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data(k=3)
        
        # Note the model is composed of <transformers encoder, (xnet), classifier>
        # We regard transformers encoder and any additional network (xnet) as feat generator
        model = self.model
        
        gen = nn.ModuleList([model.model]) # encoder
        if hasattr(model, "xnet"):
            gen.append(model.xnet)
        
        # Create discriminator
        disc = BinaryAdvClassifier(model.config.hidden_size)
        self.register_model(disc, name="disc")
        
        model_optim_config = dict()
        extended_optim_config_dict = dict()
        
        for name, lr, sched_name in [
            ("model", self.exp_args.learning_rate_task, self.exp_args.lr_scheduler_task),
            ("disc", self.exp_args.learning_rate_disc, self.exp_args.lr_scheduler_disc),
            ("gen", self.exp_args.learning_rate_gen, self.exp_args.lr_scheduler_gen)
        ]:
            optim_config = dict(
                optim="adamw", optim_kwargs={"lr": lr}, lr_scheduler=sched_name
            )
            if name == "model":
                optim_config["model"] = model
                model_optim_config = optim_config
            elif name == "disc":
                optim_config["disc"] = disc 
                extended_optim_config_dict[name] = optim_config
            elif name == "gen":
                optim_config["gen"] = gen
                extended_optim_config_dict[name] = optim_config
        
        self.model_optim_config = model_optim_config
        self.extended_optim_config_dict = extended_optim_config_dict
        self.configure_optimization()
    
    def run_epoch(self):
        model = self._models["model"]
        disc = self._models["disc"]
        
        metrics = [
            "batch_clf_loss", 
            "batch_disc_loss", "batch_disc_acc",
            "batch_gen_loss", "batch_gen_acc"
        ]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        for self.step in self.epoch_iterator:
            model.train()
            disc.train()
            
            # ==========================
            # PART-I Task optimization
            # ==========================
            src = self.get_src_batch(dataloader_idx=0)
            task_outputs = model(src)
            task_loss = task_outputs["ner"]["loss"]
            
            if self.is_update_step:
                self.global_step += 1
            
            self.model_backward_and_update(task_loss, ["model"])
            
            self.stats["batch_clf_loss"].append(self.adjust_loss(task_loss).item())
            self.stats["batch_train_loss"].append(self.adjust_loss(task_loss).item())
            
            # ==========================
            # PART-II Disc optimization
            # ==========================
            src = self.get_src_batch(dataloader_idx=1)
            tgt = self.get_tgt_batch(dataloader_idx=1)
            xs = model(src, tag=False)["encoder"]["pooled"]
            xt = model(tgt, tag=False)["encoder"]["pooled"]
            disc_outputs = disc(xs, xt, apply_grl=False, flip=False)
            disc_loss = disc_outputs["loss"]
            
            self.model_backward_and_update(disc_loss, ["disc"])
            
            self.stats["batch_disc_loss"].append(self.adjust_loss(disc_loss).item())
            if self.n_gpu > 1:
                disc_outputs["acc"] = disc_outputs["acc"].mean()
            self.stats["batch_disc_acc"].append(disc_outputs["acc"].item())
            
            # ==========================
            # PART-III Gen optimization
            # ==========================
            src = self.get_src_batch(dataloader_idx=2)
            tgt = self.get_tgt_batch(dataloader_idx=2)
            xs = model(src, tag=False)["encoder"]["pooled"]
            xt = model(tgt, tag=False)["encoder"]["pooled"]
            # Note the flip=True flag, this time we will use 0s label
            # for source and 1s for target
            gen_outputs = disc(xs, xt, apply_grl=False, flip=True)
            gen_loss = gen_outputs["loss"]
            
            self.model_backward_and_update(gen_loss, ["gen"])
            
            self.stats["batch_disc_loss"].append(self.adjust_loss(gen_loss).item())
            if self.n_gpu > 1:
                gen_outputs["acc"] = gen_outputs["acc"].mean()
            self.stats["batch_disc_acc"].append(gen_outputs["acc"].item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
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
class KeungArguments(ArgumentsBase):
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
    learning_rate_task: float = field(
        default=6e-6,
        metadata={"help": "Task learnign rate."}
    )
    lr_scheduler_task: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use."}
    )
    learning_rate_disc: float = field(
        default=5e-4,
        metadata={"help": "Discriminator learning rate."}
    )
    lr_scheduler_disc: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for discriminator."}
    )
    learning_rate_gen: float = field(
        default=6e-8,
        metadata={"help": "Generator learning rate."}
    )
    lr_scheduler_gen: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for generator."}
    )
