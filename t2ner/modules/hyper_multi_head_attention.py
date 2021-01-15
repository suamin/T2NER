# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHAParameterGenetrator(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.in_hparam_w = nn.Parameter(torch.empty(input_dim, 3*hidden_dim))
        self.in_hparam_b = nn.Parameter(torch.empty(input_dim, 3*hidden_dim))
        self.in_cross_v = nn.Parameter(torch.empty(input_dim, hidden_dim))
        
        self.out_hparam_w = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.out_hparam_b = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.out_cross_v = nn.Parameter(torch.empty(input_dim, hidden_dim))
        
        print(sum(p.numel() for p in self.parameters()))
    
    def forward(self, h):
        # h = hidden_dim
        params = dict()
        
        # generate params of in_proj(_weight/bias)
        in_h1 = h.matmul(self.in_hparam_w) # 3h
        in_h2 = h.matmul(self.in_cross_v) # h
        params["in_proj_weight"] = in_h1.unsqueeze(1).matmul(in_h2.unsqueeze(0)) # 3h x h
        params["in_proj_bias"] = h.matmul(self.in_hparam_b) # 3h
        
        # generate params of out_proj(_weight/bias)
        out_h1 = h.matmul(self.out_hparam_w) # h
        out_h2 = h.matmul(self.out_cross_v) # h
        params["out_proj_weight"] = out_h1.unsqueeze(1).matmul(out_h2.unsqueeze(0)) # h x h
        params["out_proj_bias"] = h.matmul(self.out_hparam_b) # h 
        
        return params


class MHAParameterGenetratorNetwork(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.mha_pg = MHAParameterGenetrator(input_dim, hidden_dim)
        self.init_params()
    
    def init_params(self):
        for weight in self.parameters():
            weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, h):
        h_params = self.mha_pg(h)
        return h_params


class HyperMultiHeadAttention(nn.Module):
    
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, params, mask=None):
        h = x.transpose(0, 1)
        h_ = F.multi_head_attention_forward(
            h, h, h, self.hidden_dim, self.num_heads, 
            params["in_proj_weight"], params["in_proj_bias"], 
            None, None, False, 0., params["out_proj_weight"], 
            params["out_proj_bias"], training=self.training, 
            key_padding_mask=(mask == 0), need_weights=True
        )[0]
        h = h + self.dropout(h_)
        h = h.transpose(0, 1)
        return h


def test():
    input_dim = 8
    hidden_dim = 768
    hn_mha = MHAParameterGenetratorNetwork(input_dim, hidden_dim)
    h = torch.randn(input_dim)
    h_params = hn_mha(h)
    x = torch.randn(3, 5, 768)
    mask = torch.ones(3, 5)
    h_mha = HyperMultiHeadAttention(hidden_dim, num_heads=8, dropout=0.1)
    output = h_mha(x, h_params, mask)


if __name__=="__main__":
    test()
