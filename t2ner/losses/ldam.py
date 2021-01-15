# -*- coding: utf-8 -*-

# Modified from: https://raw.githubusercontent.com/kaidic/LDAM-DRW/master/losses.py

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from ..data import IGNORE_INDEX


class MaskedSeqLDAMLoss(nn.Module):
    
    def __init__(self, num_labels, cls_num_list, max_m=0.5, weight=None, s=30):
        super().__init__()
        assert len(cls_num_list) == num_labels
        assert s > 0
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.num_labels = num_labels
        self.m_list = nn.Parameter(torch.from_numpy(m_list).float(), requires_grad=False)
        self.s = s
        self.weight = weight
    
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
        
        index = torch.zeros_like(active_logits).byte()
        temp_labels = torch.empty_like(active_labels)
        temp_labels.copy_(active_labels.data)
        temp_labels[temp_labels == IGNORE_INDEX] = 0
        index.scatter_(1, temp_labels.view(-1, 1), 1)
        index = index * (active_labels.data.view(-1, 1) != IGNORE_INDEX)
        
        index_float = index.float().to(logits.device)
        m_list = self.m_list.to(logits.device)
        batch_m = m_list[None, :].matmul(index_float.transpose(0,1)).view((-1, 1))
        active_logits_m = active_logits - batch_m
        
        active_logits = self.s * torch.where(index, active_logits_m, active_logits)
        loss = F.cross_entropy(
            active_logits,
            active_labels,
            reduction="none",
            weight=self.weight,
            ignore_index=IGNORE_INDEX
        )
        seq_len = logits.shape[1]
        loss = loss.view(-1, seq_len).sum(1).mean()
        
        return loss
