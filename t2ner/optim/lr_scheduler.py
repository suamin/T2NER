"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import torch

from transformers import optimization as optim

AVAI_SCHEDS = ["constant", "linear", "cosine", "cosinehr"]


def build_lr_scheduler(optimizer, lr_scheduler="linear", **kwargs):
    """A function wrapper for building a learning rate scheduler."""
    
    num_training_steps = kwargs.get("num_training_steps", -1)
    warmup_steps = kwargs.get("warmup_steps", 0)
    
    if lr_scheduler not in AVAI_SCHEDS:
        raise ValueError(
            "Unsupported scheduler: {}. Must be one of {}".format(
                lr_scheduler, AVAI_SCHEDS
            )
        )
    
    if lr_scheduler != "constant":
        if num_training_steps < 0 or not isinstance(num_training_steps, int):
            raise ValueError(
                "For non-constant lr schedulers, positive integer value "
                "is expected as num_training_steps."
            )    
        if 0. < warmup_steps < 1.:
            warmup_steps = int(num_training_steps * warmup_steps)
    else:
        if 0. < warmup_steps < 1. and (
            num_training_steps < 0 or not isinstance(num_training_steps, int)
        ):
            raise ValueError(
                "For constant lr scheduler with fractional warmup_steps "
                "num_training_steps is required as positive integer value."
            )
    
    if lr_scheduler == "constant":
        if warmup_steps > 0:
            scheduler = optim.get_constant_schedule_with_warmup(optimizer, warmup_steps)
        else:
            scheduler = optim.get_constant_schedule(optimizer)
    
    elif lr_scheduler == "linear":
        scheduler = optim.get_linear_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps
        )
    
    elif lr_scheduler == "cosine":
        num_cycles = kwargs.get("num_cycles", 0.5)
        scheduler = optim.get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps, num_cycles
        )
    
    elif lr_scheduler == "cosinehr":
        num_cycles = kwargs.get("num_cycles", 1.0)
        scheduler = optim.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, num_training_steps, num_cycles
        )
    
    return scheduler
