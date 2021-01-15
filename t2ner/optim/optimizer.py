"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import warnings
import torch
import torch.nn as nn

from .radam import RAdam
from transformers.optimization import AdamW

AVAI_OPTIMS = ["adam", "adamw", "amsgrad", "sgd", "rmsprop", "radam"]


def build_optimizer(model, optim="adamw", **kwargs):
    """A function wrapper for building an optimizer."""

    lr = kwargs.get("lr", 2e-5)
    weight_decay = kwargs.get("weight_decay", 0.0)
    momentum = kwargs.get("momentum", 0.9)
    sgd_dampening = kwargs.get("sgd_dampening", 0)
    sgd_nesterov = kwargs.get("sgd_nesterov", False)
    rmsprop_alpha = kwargs.get("rmsprop_alpha", 0.99)
    adam_beta1 = kwargs.get("adam_beta1", 0.9)
    adam_beta2 = kwargs.get("adam_beta2", 0.999)
    adam_epsilon = kwargs.get("adam_epsilon", 1e-8)
    staged_lr = kwargs.get("staged_lr", False)
    new_layers = kwargs.get("new_layers", ())
    base_lr_mult = kwargs.get("base_lr_mult", 0.1)
    
    if optim not in AVAI_OPTIMS:
        raise ValueError(
            "Unsupported optim: {}. Must be one of {}".format(
                optim, AVAI_OPTIMS
            )
        )
    
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model given to build_optimizer must be an instance of nn.Module"
        )
    
    if staged_lr:
        if isinstance(new_layers, str):
            if new_layers is None:
                warnings.warn(
                    "new_layers is empty, therefore, staged_lr is useless"
                )
            new_layers = [new_layers]
        
        if isinstance(model, nn.DataParallel):
            model = model.module
        
        base_params = []
        base_layers = []
        new_params = []
        
        for name, module in model.named_children():
            if name in new_layers:
                new_params += module.named_parameters()
            else:
                base_params += module.named_parameters()
                base_layers.append(name)
        
        base_decay_params, base_no_decay_params = decouple_no_decay_params(base_params)
        new_decay_params, new_no_decay_params = decouple_no_decay_params(new_params)
        param_groups = [
            {
                "params": base_decay_params,
                "weight_decay": weight_decay,
                "lr": lr * base_lr_mult
            },
            {
                "params": base_no_decay_params,
                "weight_decay": 0.0,
                "lr": lr * base_lr_mult
            },
            {
                "params": new_decay_params,
                "weight_decay": weight_decay
            },
            {
                "params": new_no_decay_params,
                "weight_decay": 0.0
            }
        ]
    
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        params = model.named_parameters()
        decay_params, no_decay_params = decouple_no_decay_params(params)
        param_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]
    
    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
        )
    
    elif optim == "adamw":
        optimizer = AdamW(
            param_groups, 
            lr=lr,
            eps=adam_epsilon,
            betas=(adam_beta1, adam_beta2)
        )
    
    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )
    
    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )
    
    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            alpha=rmsprop_alpha,
        )
    
    elif optim == "radam":
        optimizer = RAdam(
            param_groups,
            lr=lr,
            betas=(adam_beta1, adam_beta2)
        )
    
    return optimizer


def decouple_no_decay_params(params):
    no_decay = ["bias", "LayerNorm.weight"]
    decay_params = [p for n, p in params if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in params if any(nd in n for nd in no_decay)]
    return decay_params, no_decay_params
