# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os

from . import base
from .. import modules
from . import HF_MODELS, bert, xlm, xlmr

from transformers import AutoConfig, PretrainedConfig
from collections import OrderedDict


class TokenClassificationUtils(base.TransformersUtils):
    
    def __init__(self):
        super().__init__()
    
    def add_classifier(self):
        self.classifier = modules.TokenClassifier(
            self.config.hidden_size,
            self.config.num_labels,
            use_crf=self.config.use_crf,
            bigram=self.config.crf_bigram,
            loss_fct=self.config.loss_fct,
            class_num_list=self.config.class_num_list,
            ignore_bias=self.config.ignore_bias_clf,
            normalize=self.config.normalize_clf,
            temp=self.config.temp_clf
        )
        self.output_dropout = nn.Dropout(self.config.output_classifier_dropout)
    
    def forward(self, inputs, tag=True):
        input_ids = inputs["input_ids"]
        input_mask = inputs.get("input_mask", None)
        encoded = self.encode(input_ids, input_mask)
        sequence_output, cls_output, pooled, x_output = encoded
        
        if hasattr(self, "xnet"):
            sequence_output = self.xnet(encoded, mask=input_mask)
        
        sequence_output = self.output_dropout(sequence_output)
        
        if hasattr(self, "pooler"):
            pooled = self.pooler(sequence_output, mask=input_mask)
        else:
            pooled = self.output_dropout(pooled)
        
        outputs = {
            "encoder": {
                "seq": sequence_output,
                "cls": cls_output,
                "pooled": pooled,
                "x": x_output
            }
        }
        if tag:
            labels = inputs.get("label_ids", None)
            outputs["ner"] = self.classifier(sequence_output, mask=input_mask, labels=labels)
        
        return outputs


class XForTokenClassificationBase(TokenClassificationUtils):
    
    def __init__(self):
        super().__init__()
    
    def setup(self, config):
        check = sum([isinstance(config, hf_model.Config) for hf_model in HF_MODELS])
        
        # TODO add more informative message
        if not check:
            raise ValueError("Unknown / unsupported config.")
        
        for hf_model in HF_MODELS:
            if isinstance(config, hf_model.Config):
                encoder = hf_model.Model(config)
                
                # NOTE this function dynamically creates an attribute of (any sub-)class 
                # with same name as ``config.model_type`` so the weights can be correctly 
                # initialized when using the ``from_pretrained`` method. Creating an attr
                # , e.g., self.encoder will not allow the weights to be loaded. This is by 
                # design from HF library.
                self.configure_encoder(encoder)
        
        self.add_classifier()


class BertForTokenClassification(bert.PreTrainedModel, XForTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


class XLMRobertaForTokenClassification(bert.PreTrainedModel, XForTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


class XLMForTokenClassification(xlm.PreTrainedModel, XForTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (xlm.Config, XLMForTokenClassification),
        (xlmr.Config, XLMRobertaForTokenClassification),
        (bert.Config, BertForTokenClassification)
    ]
)


class AutoModelForTokenClassification:
    
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForTokenClassification is designed to be instantiated "
            "using the `AutoModelForTokenClassification.from_pretrained(pret"
            "rained_model_name_or_path)`."
        )
    
    @classmethod
    def get_models_map(cls, external_map=None):
        models_map = OrderedDict(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING)
        if external_map:
            for model_config, model_class in external_map.items():
                models_map[model_config] = model_class
        return models_map
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        
        external_map = kwargs.pop("models_map", None)
        models_map = cls.get_models_map(external_map)
        
        for config_class, model_class in models_map.items():
            if isinstance(config, config_class):
                
                # FIXME HF default config forcibly sets `id2label` with 2 labels;
                # to ovrride pass a custom map in config_kwargs. There is no easy
                # way to check this in `check_config_for_specific_attr` for example
                # except to override variable name or change this behavior in
                # ``PretrainedConfig`` class, which is unnecessary.
                config_kwargs = kwargs.pop("config_kwargs", dict())
                
                # In case we have a trained model and we are transfering from
                # dataset with different number of entity types, the classifier
                # weights have to be explicitly removed to avoid shape mismatch
                # when using `from_pretrained`.
                state_dict = None
                if os.path.isdir(pretrained_model_name_or_path):
                    ignore_heads = config_kwargs.get("ignore_heads", False)
                    if ignore_heads:
                        weights_file = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                        state_dict = torch.load(weights_file, map_location="cpu")
                        for k in list(state_dict.keys()):
                            if k.startswith("classifier"):
                                del state_dict[k]
                
                config.update(config_kwargs)
                
                model_class.check_config_for_shared_attr(config)
                model_class.check_config_for_specific_attr(config)
                
                return model_class.from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    config=config, 
                    state_dict=state_dict, 
                    **kwargs
                )
        
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.keys()),
            )
        )
