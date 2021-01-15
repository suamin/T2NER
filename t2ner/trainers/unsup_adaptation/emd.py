# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd

from dataclasses import dataclass, field

from ...base import ArgumentsBase
from ...modules import WassersteinCritic, SeqWassersteinCritic
from ..adaptation import AdaptationTrainer


class EMDTrainer(AdaptationTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
    
    def before_train(self):
        self.load_train_data(k=2)
        model = self.model
        
        # Create critic
        if self.exp_args.pooling is None:
            critic = SeqWassersteinCritic(
                model.config.hidden_size, 
                num_layers=self.exp_args.num_critic_layers, 
                dropout=self.exp_args.critic_dropout
            )
            self.pool = False
        else:
            critic = WassersteinCritic(
                model.config.hidden_size, 
                num_layers=self.exp_args.num_critic_layers, 
                dropout=self.exp_args.critic_dropout
            )
            self.pool = True
        self.register_model(critic, name="critic")
        
        model_optim_config = dict()
        extended_optim_config_dict = dict()
        
        for name, lr, sched_name in [
            ("model", self.training_args.learning_rate, self.exp_args.lr_scheduler),
            ("critic", self.exp_args.learning_rate_critic, self.exp_args.lr_scheduler_critic)
        ]:
            optim_config = dict(
                optim="adamw", optim_kwargs={"lr": lr}, lr_scheduler=sched_name
            )
            if name == "model":
                optim_config["model"] = model
                model_optim_config = optim_config
            elif name == "critic":
                optim_config["critic"] = critic 
                extended_optim_config_dict[name] = optim_config
        
        self.model_optim_config = model_optim_config
        self.extended_optim_config_dict = extended_optim_config_dict
        self.configure_optimization()
    
    def gradient_penalty(self, xs, xt, critic, masks=None, maskt=None):
        """Modified from: 
        
        https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L129
        
        """
        if self.pool:
            eps = torch.rand(xs.shape[0], 1).to(self.device)
            # linearly interpolated pairs
            inputs = eps * xs + ((1 - eps) * xt)
            inputs = autograd.Variable(inputs, requires_grad=True)
            outputs = critic(inputs)["logits"]
        else:
            eps = torch.rand(xs.shape[:-1]).unsqueeze(-1).to(self.device)
            if masks is not None and maskt is not None:
                mask = masks * maskt
            else:
                mask = None
            inputs = eps * xs + ((1 - eps) * xt)
            inputs = autograd.Variable(inputs, requires_grad=True)
            outputs = critic(inputs, mask=mask)["logits"]  
        
        grad_outputs = torch.ones(outputs.size()).to(self.device)
        grad = autograd.grad(
            inputs=inputs, outputs=outputs, grad_outputs=grad_outputs, 
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_norm = grad.norm(2, dim=-1)
        
        if not self.pool:
            if mask is not None:
                grad_pen = ((grad_norm - 1) * mask) ** 2
            else:
                grad_pen = (grad_norm - 1) ** 2
            grad_pen = grad_pen.sum(1)
        else:
            grad_pen = (grad_norm - 1) ** 2
        
        grad_pen = grad_pen.mean()
        
        return grad_pen
    
    def critic_half_score(self, encoded, batch, critic):
        if self.pool:
            x = encoded["pooled"]
            c = critic(x)
        else:
            x = encoded["seq"]
            mask = batch.get("input_mask", None)
            c = critic(x, mask=mask)
        return x, c
    
    def emd_train_step(self, model, critic):
        self.freeze(model)
        self.unfreeze(critic)
        
        # critic iterations
        for _ in range(self.exp_args.n_critic):
            src = self.get_src_batch(dataloader_idx=0)
            encodeds = model(src, tag=False)["encoder"]
            xs, cs = self.critic_half_score(encodeds, src, critic)
            loss_qs = cs["loss"]
            self.model_backward(-loss_qs)
            
            tgt = self.get_tgt_batch(dataloader_idx=0)
            encodedt = model(tgt, tag=False)["encoder"]
            xt, ct = self.critic_half_score(encodedt, tgt, critic)
            loss_qt = ct["loss"]
            
            if self.exp_args.use_gp:
                gp = self.gradient_penalty(
                    xs.data, xt.data, critic, 
                    src.get("input_mask", None),
                    tgt.get("input_mask", None)
                )
                loss_qt += self.exp_args.gp_lmbda*gp
            
            self.model_backward(loss_qt)
            self.model_update(["critic"])
            self.update_lr(["critic"])
            self.model_zero_grad(["critic"])
            
            if not self.exp_args.use_gp:
                # clip critic weight to enforce Lipschitz constraint
                for p in critic.parameters():
                    p.data.clamp_(-self.exp_args.critic_clamp, self.exp_args.critic_clamp)
        
        # FIXME if model is not in 'finetune', i.e., parts of model
        # are frozen by user then this will unfreeze them?
        self.unfreeze(model)
        self.freeze(critic)
        
        src = self.get_src_batch(dataloader_idx=1)
        tgt = self.get_tgt_batch(dataloader_idx=1)
        
        self.model_zero_grad(["model", "critic"])
        
        outputs = model(src)
        clf_loss = outputs["ner"]["loss"]
        self.model_backward(clf_loss, retain_graph=True)
        
        encodeds = outputs["encoder"]
        xs, cs = self.critic_half_score(encodeds, src, critic)
        loss_qs = cs["loss"]
        self.model_backward(self.exp_args.lmbda * loss_qs, retain_graph=True)
        
        encodedt = model(tgt, tag=False)["encoder"]
        xt, ct = self.critic_half_score(encodedt, tgt, critic)
        loss_qt = ct["loss"]
        if self.exp_args.use_gp:
            gp = self.gradient_penalty(
                xs.data, xt.data, critic,
                src.get("input_mask", None),
                tgt.get("input_mask", None)
            )
            loss_qt = loss_qt + self.exp_args.gp_lmbda*gp
        
        self.model_backward(-self.exp_args.lmbda * loss_qt)
        
        if not self.exp_args.use_gp:
            # clip critic weight to enforce Lipschitz constraint
            for p in critic.parameters():
                p.data.clamp_(-self.exp_args.critic_clamp, self.exp_args.critic_clamp)
                # Q.net[1].weight.grad.data.norm()
        
        critic_loss = loss_qs - loss_qt
        
        return clf_loss, critic_loss
    
    def run_epoch(self):
        model = self._models["model"]
        critic = self._models["critic"]
        
        metrics = ["batch_clf_loss", "batch_critic_loss"]
        for k in metrics:
            if k not in self.stats:
                self.stats[k] = list()
        
        for self.step in self.epoch_iterator:
            clf_loss, critic_loss = self.emd_train_step(model, critic)
            loss = clf_loss + self.exp_args.lmbda*critic_loss
            
            self.stats["batch_train_loss"].append(self.adjust_loss(loss).item())
            self.stats["batch_clf_loss"].append(self.adjust_loss(clf_loss).item())
            self.stats["batch_critic_loss"].append(self.adjust_loss(critic_loss).item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
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
class EMDArguments(ArgumentsBase):
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
        default="constant",
        metadata={"help": "Which learning rate scheduler to use."}
    )
    learning_rate_critic: float = field(
        default=5e-4,
        metadata={"help": "Discriminator learning rate."}
    )
    lr_scheduler_critic: str = field(
        default="constant",
        metadata={"help": "Which learning rate scheduler to use for discriminator."}
    )
    num_critic_layers: int = field(
        default=1,
        metadata={"help": "No. of in MLP classifier (default = one linear layer)."}
    )
    critic_dropout: float = field(
        default=0.,
        metadata={"help": "Dropout value in MLP layers of critic."}
    )
    n_critic: int = field(
        default=10,
        metadata={"help": "No. of critic (disc) training steps in Wasserstein setup."}
    )
    use_gp: bool = field(
        default=False,
        metadata={"help": "Use gradient penalty instead of weight clipping to enforce Lipschitz constraint."}
    )
    critic_clamp: float = field(
        default=0.01,
        metadata={"help": "Clip values of disc parameters in Wasserstein setup."}
    )
    gp_lmbda: float = field(
        default=10.,
        metadata={"help": "Trade-off parameter for gradient penalty."}
    )
    lmbda: float = field(
        default=0.1,
        metadata={"help": "Trade-off parameter between task and adversarial loss."}
    )
