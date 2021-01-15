# -*- coding: utf-8 -*-

import torch.nn as nn


class Transformer(nn.Module):
    
    def __init__(
        self,
        input_dim,
        hidden_dim=768,
        num_layers=1,
        num_heads=8,
        dropout=0.1,
        activation="relu"
    ):
        super().__init__()
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            input_dim,
            num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
        )
        self.transformer = nn.TransformerEncoder(transformer_encoder_layer, num_layers)
    
    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2]).to(x.device)
        h = self.transformer(x.transpose(0, 1), src_key_padding_mask=(mask == 0))
        h = h.transpose(0, 1)
        return h
