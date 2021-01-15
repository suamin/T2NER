# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class MaskedSoftmax(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, mask=None, dim=-1, epsilon=1e-5):
        # cf. https://discuss.pytorch.org/t/apply-mask-softmax/14212/15
        exps = torch.exp(logits)
        if mask is not None:
            masked_exps = exps * mask.unsqueeze(-1).float()
        else:
            masked_exps = exps
        masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
        return masked_exps / masked_sums
