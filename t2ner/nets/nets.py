# -*- coding: utf-8 -*-

import torch.nn as nn

from ..modules import LSTM, Transformer


class XNetBase(nn.Module):
    
    def __init__(self, config):
        super().__init__()
    
    def forward(self, encoded, mask=None):
        raise NotImplementedError


class LSTMNet(XNetBase):
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.dropout = nn.Dropout(kwargs.get("input_dropout", config.hidden_dropout_prob))
        self.xlstm = LSTM(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size // 2,
            num_layers=kwargs.get("num_layers", 2)
        )
    
    def forward(self, encoded, mask=None):
        x = encoded[0]
        x = self.dropout(x)
        h = self.xlstm(x, mask)
        return h


class TransformerNet(XNetBase):
    
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.dropout = nn.Dropout(kwargs.get("input_dropout", config.hidden_dropout_prob))
        self.xtrsf = Transformer(
            input_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            num_layers=kwargs.get("num_layers", 2),
            num_heads=kwargs.get("num_heads", config.num_attention_heads),
            dropout=kwargs.get("hidden_dropout", config.hidden_dropout_prob),
            activation="relu"
        )
    
    def forward(self, encoded, mask=None):
        x = encoded[0]
        x = self.dropout(x)
        h = self.xtrsf(x, mask)
        return h
