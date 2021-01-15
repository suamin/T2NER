# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class CausalLanguageModelingLoss(nn.Module):
    
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        # labels are assumed to be input_ids; as we are 
        # doing next-token prediction; shift prediction 
        # scores and input ids by one
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss = self.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return loss
