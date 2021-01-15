# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from ..losses import CausalLanguageModelingLoss


class TransformersCLM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size)
        )
        
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
    
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() != 2:
            raise ValueError(
                "Expected 2-dim (batch_size x seq_length) mask."
            )
        
        batch_size, seq_length = attention_mask.shape # batch_size x seq_len
        
        # Create mask for causal language modeling
        mask = (
            torch.triu(torch.ones(seq_length, seq_length))
            .transpose(0, 1)[None, :, :]
            .repeat(batch_size, 1, 1)
        )
        mask = mask.to(attention_mask.device) # batch_size x seq_len x seq_len
        extended_attention_mask = mask[:, None, :, :] * attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(dtype=attention_mask.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, model, input_ids, attention_mask=None):
        embedding_output = model.embeddings(input_ids=input_ids)
        head_mask = model.get_head_mask(None, self.num_hidden_layers)
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=input_ids.device)
        
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        encoder_outputs = model.encoder(
            embedding_output, 
            attention_mask=extended_attention_mask,
            head_mask=head_mask
        )
        sequence_output = encoder_outputs[0]
        prediction_scores = self.net(sequence_output)
        loss = CausalLanguageModelingLoss(self.vocab_size)(prediction_scores, input_ids)
        outputs = {"logits": prediction_scores, "loss": loss}
        
        return outputs
