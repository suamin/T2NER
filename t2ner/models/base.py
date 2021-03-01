# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import dataclasses

from dataclasses import dataclass, field

from .. import modules
from .. import nets

from ..base import ArgumentsBase


class TransformersUtils(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    @classmethod
    def check_config_for_shared_attr(cls, config):
        update_kwargs = dict()
        
        # `mode` allows to specify usage of transformers encoder:
        #
        # 'finetune': fine-tune the encoder during training and 
        #             use output hidden states of last layer.
        # 'freeze':   freeze the encoder during training and use 
        #             output hidden states of last last.
        # 'firstn': freeze first n layers of the encoder (note 
        #             that layer 0 is embedding).
        # 'lastn': freeze the encoder and use the summation of last 
        #             n layers' hidden states.
        # 'embedonly': use only word embedding for encoding.
        
        # TODO 
        # 1. add references to the papers
        # 2. make `n` configurable in the first-n/last-n modes  
        
        if not hasattr(config, "mode"):
            update_kwargs["mode"] = "finetune"
        else:
            mode = config.mode
            
            if mode.startswith("first"):
                n = int(mode[5:])
                update_kwargs["firstn"] = n
            
            elif mode.startswith("last"):
                n = int(mode[4:])
                update_kwargs["lastn"] = n
                update_kwargs["output_hidden_states"] = True
            
            elif mode not in (
                "finetune", "freeze", "embedonly"
            ):
                # fallback to 'finetune' in case of invalid options
                update_kwargs["mode"] = "finetune"
        
        if not hasattr(config, "output_classifier_dropout"):
            update_kwargs["output_classifier_dropout"] = 0.1
        
        if not hasattr(config, "output_attentions"):
            update_kwargs["output_attentions"] = True
        
        if not hasattr(config, "add_xnet"):
            update_kwargs["add_xnet"] = None
        else:
            if config.add_xnet == "lstm":
                self.xnet_class = nets.LSTMNet
            elif config.add_xnet == "transformer":
                self.xnet_class = net.TransformerNet
        
        if not hasattr(config, "loss_fct"):
            update_kwargs["loss_fct"] = "ce"
        
        if not hasattr(config, "class_num_list"):
            update_kwargs["class_num_list"] = None
        
        if not hasattr(config, "use_crf"):
            update_kwargs["use_crf"] = False
        
        if not hasattr(config, "crf_bigram"):
            update_kwargs["crf_bigram"] = None
        
        if not hasattr(config, "ignore_bias_clf"):
            update_kwargs["ignore_bias_clf"] = False
        
        if not hasattr(config, "normalize_clf"):
            update_kwargs["normalize_clf"] = False
        
        if not hasattr(config, "temp_clf"):
            update_kwargs["temp_clf"] = -1
        
        if not hasattr(config, "ignore_heads"):
            update_kwargs["ignore_heads"] = False
        
        config.update(update_kwargs)
    
    @classmethod
    def check_config_for_specific_attr(cls, config):
        pass
    
    def configure_encoder(self, model):
        setattr(self, self.config.model_type, model)
        if self.config.mode == "freeze" or self.config.mode.startswith("last"):
            self.freeze(self.model)
        elif self.config.mode.startswith("first"):
            self.freeze_layers(self.config.firstn)
    
    @property
    def model(self):
        return getattr(self, self.config.model_type)
    
    @staticmethod
    def freeze(module):
        for params in module.parameters():
            params.requires_grad = False
    
    @staticmethod
    def unfreeze(module):
        for params in module.parameters():
            params.requires_grad = True
    
    def freeze_layers(self, n=-1):
        # Freeze upto first n layers (n=0 embedding)
        # cf. https://arxiv.org/pdf/1904.09077.pdf
        if n <= 0: return
        for i in range(n):
            if i == 0: self.freeze(self.model.embeddings)
            else: self.freeze(self.model.encoder.layer[i - 1])
    
    def encode(self, input_ids, input_mask=None):
        if self.config.mode == "embedonly":
            embeddings = self.model.get_input_embeddings()
            sequence_output = embeddings(input_ids)
            cls_output = sequence_output[:, 0, :]
            pooled = None
            x_output = None
        else:
            # TODO check if this interface works for all encoders
            # (it is limited, e.g., one may want to pass ``langs``
            # data when using XLM)
            outputs = self.model(input_ids, attention_mask=input_mask)
            if self.config.mode.startswith("last"):
                sequence_output = torch.stack(outputs[2]) # num_layers x [B x L x E]
                sequence_output = sequence_output[-self.config.lastn:].sum(0)
            else:
                sequence_output = outputs[0]
            cls_output = sequence_output[:, 0, :]
            pooled =  outputs[1]
            x_output = outputs[2:]
        return sequence_output, cls_output, pooled, x_output
    
    def add_xnet(self, xnet):
        self.xnet = xnet
    
    def add_pooler(self, method="mean"):
        if method == "attn":
            self.pooler = modules.Pooling(method, hidden_dim=self.config.hidden_size)
        else:
            self.pooler = modules.Pooling(method)


@dataclass
class ModelArguments(ArgumentsBase):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    do_lower_case: bool = field(
        default=False,
        metadata={"help": "Whether to lowercase the data."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    features_cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to save the features."}
    )
    overwrite_cache: bool = field(
        default=False, 
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    encoder_mode: str = field(
        default="finetune",
        metadata={
            "help":
            "This option defines the mode of base HF transformers model for encoding "
            "the input with options ['finetune', 'freeze', 'lastn', 'firstn', "
            "'embedonly']. By default the model is finetuned during training."
        }
    )
    ignore_heads: bool = field(
        default=False,
        metadata={"help": "If loading a trained model with classification heads, ignore them."}
    )
    add_xnet: str = field(
        default=None,
        metadata={"help": "Add a 2-layer BiLSTM or 2-layer transformers after encoder."}
    )
    output_classifier_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout value to use before classifier."}
    )
    loss_fct: float = field(
        default="ce",
        metadata={"help": "Loss function for classification (default is cross-entropy)."}
    )
    valid_metric: str = field(
        default="f1",
        metadata={"help": "Validation metric for recording best models (f1 or loss)."}
    )
    class_num_list: list = field(
        default=None,
        metadata={"help": "List of no. of class instances in training data."}
    )
    ignore_bias_clf: bool = field(
        default=False,
        metadata={"help": "Ignore bias term in linear output classifier."}
    )
    normalize_clf: bool = field(
        default=False,
        metadata={"help": "Apply L2-normalization to outputs of linear output classifier."}
    )
    temp_clf: float = field(
        default=-1,
        metadata={"help": "Temperature scaling of outputs of linear output classifier."}
    )
    use_crf: bool = field(
        default=False,
        metadata={"help": "Whether to use CRF prediction layer."}
    )
    crf_bigram: bool = field(
        default=False,
        metadata={"help": "If using CRF, what order (default is 0th-order)."}
    )
    cache_dir: str = field(
        default=None, 
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3."}
    )
    init_checkpoint: str = field(
        default=None,
        metadata={"help": "Initial checkpoint for train/predict."}
    )
