# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class Pooling(nn.Module):
    
    def __init__(self, mode="mean", hidden_dim=None):
        super().__init__()
        self.mode = mode
        if mode == "attn":
            self.linear = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, x, mask=None, weights=None, normalize_weights=False):
        if mask is None:
            mask = torch.ones(x.shape[:2]).to(x.device)
        
        if self.mode == "attn":
            batch_size, seq_len, hidden_dim = x.shape
            h = x.contiguous().view(-1, hidden_dim)
            attn_scores = self.linear(h).view(batch_size, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
            attn = torch.softmax(attn_scores, dim=1)
            output = attn.unsqueeze(1).bmm(x).squeeze(1)
        
        else:
            if weights is not None:
                if normalize_weights:
                    weights = weights.masked_fill(mask == 0, -1e9)
                    weights = torch.softmax(weights, dim=1)
                else:
                    weights = weights.masked_fill(mask == 0, 0.)
            else:
                weights = torch.ones(x.shape[:2]).float().to(x.device)
            
            x = x * weights.unsqueeze(-1)
            mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
            
            if self.mode == "max":
                output = torch.max(x, 1)[0]
            else:
                sum_x = torch.sum(x * mask_expanded, 1)
                if self.mode == "sum":
                    output = sum_x
                else:
                    sum_mask = mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    if self.mode == "mean_sqrt_len":
                        output = sum_x / torch.sqrt(sum_mask)
                    else:
                        output = sum_x / sum_mask    
        
        return output
