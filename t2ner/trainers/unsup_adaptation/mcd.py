# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import TokenClassifier, MaskedSoftmax
from ..adaptation import AdaptationTrainer


class MCDTrainer(AdaptationTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data()

        # Note the model is composed of <trasnformers encoder, (xnet), classifier 1>
        # This is main classifier that is used for inference as well. We regard
        # trasnformers encoder and any additional network (xnet) as features generator.
        model = self.model
        
        try:
            assert not model.config.use_crf
            assert model.config.loss_fct == "ce"
        except:
            raise NotImplementedError
        
        G = nn.ModuleList([model.model]) # encoder
        if hasattr(model, "xnet"):
            G.append(model.xnet)
        
        # Two classifiers (discriminators)
        C1 = model.classifier
        C2 = TokenClassifier(model.config.hidden_size, model.config.num_labels)
        
        self.register_model(C1, name="C1")
        self.register_model(C2, name="C2")
        
        model_optim_config = dict()
        extended_optim_config_dict = dict()
        
        for name, lr, sched_name in [
            ("model", self.exp_args.learning_rate_F, self.exp_args.lr_scheduler_F),
            ("C1", self.exp_args.learning_rate_C1, self.exp_args.lr_scheduler_C1),
            ("C2", self.exp_args.learning_rate_C2, self.exp_args.lr_scheduler_C2)
        ]:
            optim_config = dict(
                optim="adamw", optim_kwargs={"lr": lr}, lr_scheduler=sched_name
            )
            if name == "model":
                optim_config["model"] = G
                model_optim_config = optim_config
            
            elif name == "C1":
                optim_config["C1"] = C1
                extended_optim_config_dict[name] = optim_config
            
            elif name == "C2":
                optim_config["C2"] = C2
                extended_optim_config_dict[name] = optim_config
        
        self.model_optim_config = model_optim_config
        self.extended_optim_config_dict = extended_optim_config_dict
        self.configure_optimization()
    
    def discrepancy(self, y1, y2):
        # y{1,2} = batch_size x seq_len x num_classes
        return (y1 - y2).abs().sum(1).mean()
    
    def run_epoch(self):
        model = self._models["model"]
        C1 = self._models["C1"]
        C2 = self._models["C2"]
        softmax = MaskedSoftmax()
        
        metrics = ["batch_step_A_loss", "batch_step_B_loss", "batch_step_C_loss"]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        for self.step in self.epoch_iterator:
            model.train()
            C1.train()
            C2.train()
            
            src, tgt = self.get_src_batch(), self.get_tgt_batch()
            
            # ==========================
            # Step A
            # ==========================
            xs = model(src, tag=False)["encoder"]["seq"]
            mask, labels = src["input_mask"], src["label_ids"]
            C1_loss = C1(xs, mask, labels)["loss"]
            C2_loss = C2(xs, mask, labels)["loss"]
            
            loss_step_A = C1_loss + C2_loss # Eq.2
            
            if self.is_update_step:
                self.global_step += 1
            
            self.model_backward_and_update(loss_step_A)
            
            self.stats["batch_train_loss"].append(self.adjust_loss(C1_loss).item())
            self.stats["batch_step_A_loss"].append(self.adjust_loss(loss_step_A).item())
            
            # ==========================
            # Step B
            # ==========================
            with torch.no_grad():
                xs = model(src, tag=False)["encoder"]["seq"]

            C1_loss = C1(xs, mask, labels)["loss"]
            C2_loss = C2(xs, mask, labels)["loss"]
            
            with torch.no_grad():
                xt = model(tgt, tag=False)["encoder"]["seq"]
            
            mask = tgt["input_mask"]
            xt_p1 = softmax(C1(xt, mask)["logits"], mask)
            xt_p2 = softmax(C2(xt, mask)["logits"], mask)
            adv_loss = self.discrepancy(xt_p1, xt_p2)
            
            loss_step_B = (C1_loss + C2_loss) - adv_loss # Eq.4
            self.model_backward_and_update(loss_step_B, ["C1", "C2"])
            
            self.stats["batch_step_B_loss"].append(self.adjust_loss(loss_step_B).item())
            
            # ==========================
            # Step C
            # ==========================
            for _ in range(self.exp_args.n_gen):
                xt = model(tgt, tag=False)["encoder"]["seq"]
                xt_p1 = softmax(C1(xt, mask)["logits"], mask)
                xt_p2 = softmax(C2(xt, mask)["logits"], mask)
                
                loss_step_C = self.discrepancy(xt_p1, xt_p2)
                self.model_backward_and_update(loss_step_C, ["model"])
            
            self.stats["batch_step_C_loss"].append(self.adjust_loss(loss_step_C).item())
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
class MCDArguments(ArgumentsBase):
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
        default=5e-6,
        metadata={"help": "Feature generator learning rate."}
    )
    lr_scheduler_F: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for feature generator."}
    )
    learning_rate_C1: float = field(
        default=5e-6,
        metadata={"help": "Classifier 1 learning rate."}
    )
    lr_scheduler_C1: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for classifier 1."}
    )
    learning_rate_C2: float = field(
        default=5e-6,
        metadata={"help": "Classifier 2 learning rate."}
    )
    lr_scheduler_C2: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for classifier 2."}
    )
    n_gen: int = field(
        default=4,
        metadata={"help": "No. of generator iterations."}
    )
