# -*- coding: utf-8 -*-

# Modified from: https://raw.githubusercontent.com/kaidic/LDAM-DRW/master/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import IGNORE_INDEX


class MaskedSeqFocalLoss(nn.Module):
    
    def __init__(self, num_labels, weight=None, gamma=2.):
        super().__init__()
        assert gamma >= 0, "gamma should be >= 0 for focal_loss (good values in [0,5])"
        self.num_labels = num_labels
        self.weight = weight
        self.gamma = gamma
    
    def forward(self, logits, labels, mask=None):
        if mask is not None:
            active_loss = mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, 
                labels.view(-1), 
                torch.tensor(IGNORE_INDEX).type_as(labels)
            )
        else:
            active_logits = logits.view(-1, self.num_labels)
            active_labels = labels.view(-1)
        
        loss = F.cross_entropy(
            active_logits,
            active_labels,
            reduction="none",
            weight=self.weight,
            ignore_index=IGNORE_INDEX
        )
        seq_len = logits.shape[1]
        loss = focal_loss(loss, self.gamma)
        loss = loss.view(-1, seq_len).sum(1).mean()
        
        return loss


def focal_loss(cross_entropy_outputs, gamma):
    """Computes the focal loss"""
    p = torch.exp(-cross_entropy_outputs)
    loss = (1 - p) ** gamma * cross_entropy_outputs
    return loss
