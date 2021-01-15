# -*- coding: utf-8 -*-

"""
Modified from https://github.com/thespectrewithin/joint_align/blob/master/feature_ner.py

"""

import torch
import torch.nn as nn
import numpy as np

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LSTM(nn.Module):
    
    def __init__(
        self, 
        input_dim, 
        hidden_dim=384,
        num_layers=1
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True
        )
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2]).to(x.device)
        lengths = mask.sum(1).long()
        lengths, sorted_idx = lengths.sort(0, descending=True)
        x = x[sorted_idx]
        
        packed_x = pack_padded_sequence(x, lengths.cpu().data.numpy(), batch_first=True)
        
        # TODO (why it happens) current fix cf. https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011
        self.bilstm.flatten_parameters()
        
        packed_h, _ = self.bilstm(packed_x)
        h, _ = pad_packed_sequence(packed_h, batch_first=True, total_length=x.size(1))
        unsorted_idx = torch.from_numpy(np.argsort(sorted_idx.cpu().data.numpy())).to(x.device)
        h = h[unsorted_idx]
        
        return h
