# -*- coding: utf-8 -*-

import torch.nn as nn

from .crf import ChainCRF
from ..losses import MaskedSeqCrossEntropyLoss, MaskedSeqFocalLoss, MaskedSeqLDAMLoss
from ..data import IGNORE_INDEX


class Tagger(nn.Module):
    
    def __init__(
        self,
        input_dim,
        num_labels,
        use_crf=False,
        bigram=None,
        loss_fct=None,
        class_num_list=None,
        ignore_bias=False,
        normalize=False,
        temp=-1
    ):
        super().__init__()
        if use_crf:
            if bigram is None:
                bigram = True
            self.crf = ChainCRF(input_dim, num_labels, bigram=bigram)
        else:
            self.clf = nn.Linear(input_dim, num_labels, bias=not ignore_bias)
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.loss_fct = loss_fct
        self.class_num_list = class_num_list
        self.normalize = normalize
        self.temp = temp
    
    def crf_forward(self, x, mask=None, labels=None):
        if mask is not None:
            mask = mask.float()
        energy = self.crf(x, mask=mask)
        target = labels.masked_fill(labels == IGNORE_INDEX, self.num_labels)
        log_probs = energy
        outputs = {"logits": log_probs}
        
        if labels is not None:
            outputs["loss"] = self.crf.loss(log_probs, labels, mask=mask)
            # If in evaluation mode, perform decoding here
            if not self.training:
                outputs["prediction"] = self.crf.decode(log_probs, mask=mask)
        
        return outputs
    
    def clf_forward(self, x, mask=None, labels=None):
        if self.normalize:
            x = nn.functional.normalize(x, p=2, dim=-1)
        logits = self.clf(x)
        if self.temp > 0.:
            logits /= self.temp
        outputs = {"logits": logits}
        
        if labels is not None:
            if self.loss_fct == "focal":
                loss_fct = MaskedSeqFocalLoss(self.num_labels)
            elif self.loss_fct == "ldam":
                loss_fct = MaskedSeqLDAMLoss(self.num_labels, self.class_num_list)
            else:
                loss_fct = MaskedSeqCrossEntropyLoss(self.num_labels)
            outputs["loss"] = loss_fct(logits, labels, mask)
        
        return outputs
    
    def forward(self, x, mask=None, labels=None):
        if self.use_crf:
            outputs = self.crf_forward(x, mask, labels) # Use CRF
        else:
            outputs = self.clf_forward(x, mask, labels) # Use simple linear classifier
        return outputs
