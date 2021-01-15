# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os

from . import base
from .. import modules
from . import HF_MODELS, bert, xlm, xlmr

from transformers import AutoConfig, PretrainedConfig
from collections import OrderedDict


class MultiTokenClassificationUtils(base.TransformersUtils):
    
    def __init__(self):
        super().__init__()
        self.classifiers = nn.ModuleList()
        self.aux_modules = nn.ModuleList()
    
    @classmethod
    def check_config_for_specific_attr(cls, config):
        update_kwargs = dict()
        for k in ("multidata_type", "heads_info", "lang2id", "domain2id"):
            if not hasattr(config, k):
                raise AttributeError(
                    "Missing required attribute in the configuration, "
                    "{}. See data.MultiData.".format(k)
                )
            
            if not hasattr(config, "private_clf"):
                update_kwargs["private_clf"] = False
            
            if not hasattr(config, "shared_clf"):
                update_kwargs["shared_clf"] = True
            
            if k == "lang2id":
                update_kwargs["num_langs"] = len(config.lang2id)
            
            if k == "domain2id":
                update_kwargs["num_domains"] = len(config.domain2id)
        
        if not hasattr(config, "all_shared"):
            update_kwargs["all_shared"] = False
        else:
            if not hasattr(config, "num_shared_labels"):
                raise AttributeError(
                    "Missing required attribute num_shared_labels for "
                    "training with shared labels."
                )
        
        if not hasattr(config, "ignore_metadata"):
            update_kwargs["ignore_metadata"] = True
        
        if not hasattr(config, "add_lang_clf"):
            update_kwargs["add_lang_clf"] = False
        
        if not hasattr(config, "add_domain_clf"):
            update_kwargs["add_domain_clf"] = False
        
        if not hasattr(config, "add_type_clf"):
            update_kwargs["add_type_clf"] = False
        
        if not hasattr(config, "add_all_outside_clf"):
            update_kwargs["add_all_outside_clf"] = False
        
        if not hasattr(config, "add_lm"):
            update_kwargs["add_lm"] = False
        
        if not hasattr(config, "pooling"):
            update_kwargs["pooling"] = "mean"
        else:
            if not config.pooling:
                update_kwargs["pooling"] = "mean"
        
        config.update(update_kwargs)
    
    def _add_token_classifier(self, head_name, num_labels, aux=False):
        if aux:
            module_list = self.aux_modules
        else:
            module_list = self.classifiers
        module_list.add_module(
            head_name,
            modules.TokenClassifier(
                self.config.hidden_size,
                num_labels,
                use_crf=self.config.use_crf,
                bigram=self.config.crf_bigram,
                loss_fct=self.config.loss_fct
            )
        )
    
    def add_classifiers(self):
        for unique_name, head_info in self.config.heads_info.items():
            num_labels = head_info["num_labels"]
            num_types = head_info["num_types"]
            
            if self.config.private_clf:
                self._add_token_classifier(head_info["private_head_name"], num_labels)
            
            if self.config.shared_clf:
                self._add_token_classifier(head_info["shared_head_name"], num_labels)

            if self.config.add_type_clf:
                self._add_token_classifier(str(head_info["shared_head_name"]) + "_type", num_types)
            
            if not (self.config.private_clf or self.config.shared_clf):
                raise ValueError(
                    "At least a private or shared classification layer must be set."
                )
        
        if self.config.all_shared:
            self._add_token_classifier("shared", self.config.num_shared_labels)
        
        self.output_dropout = nn.Dropout(self.config.output_classifier_dropout)
    
    def add_lang_classifier(self):
        if self.config.num_langs < 2 or self.config.ignore_metadata:
            return
        
        if not self.config.add_lang_clf:
            return
        
        self.aux_modules.add_module(
            "lang_clf",
            modules.Classifier(self.config.hidden_size, self.config.num_langs)
        )
        if not hasattr(self, "pooler"):
            self.add_pooler(self.config.pooling)
        
        # if self.config.multidata_type == "MLSD" and self.config.weighted_clf:
        #     self.aux_modules.add_module(
        #         "lang_clf_weights", 
        #         nn.Parameter(self.config.num_langs)
        #     )
    
    def add_domain_classifier(self):
        if self.config.num_domains < 2 or self.config.ignore_metadata:
            return
        
        if not self.config.add_domain_clf:
            return
        
        self.aux_modules.add_module(
            "domain_clf",
            modules.Classifier(self.config.hidden_size, self.config.num_domains)
        )
        if not hasattr(self, "pooler"):
            self.add_pooler(self.config.pooling)
    
    def add_language_modeling(self):
        if not self.config.add_lm:
            return
            
        self.aux_modules.add_module(
            "lm_head",
            modules.TransformersCLM(self.config)
        )
    
    def add_all_outside_classifier(self):
        if not self.config.add_all_outside_clf:
            return
        
        self.aux_modules.add_module(
            "all_outside_clf",
            modules.Classifier(self.config.hidden_size, 2)
        )
    
    def __getitem__(self, value):
        value = str(value)
        # FIXME If value is same for some classifier and
        # auxiliary module then find a way to resolve
        try:
            module = self.classifiers.__getattr__(value)
        except AttributeError:
            module = self.aux_modules.__getattr__(value)
        return module
    
    def forward_lm(self, inputs):
        outputs = dict()
        for head, sub_inputs in inputs.items():
            input_ids = sub_inputs 
            input_mask = inputs.get("input_mask", None)
            outputs[head] = self["lm_head"](
                self.model,
                input_ids=sub_inputs["input_ids"],
                attention_mask=sub_inputs.get("input_mask", None)
            )
        return outputs
    
    def joint_encode(self, inputs):
        encoded = dict()
        
        for head, sub_inputs in inputs.items():
            input_ids = sub_inputs["input_ids"]
            input_mask = sub_inputs.get("input_mask", None)
            sub_encoded = self.encode(input_ids, input_mask)
            
            sequence_output, cls_output, pooled, x_output = sub_encoded
            
            if hasattr(self, "xnet"):
                sequence_output = self.xnet(sub_encoded, mask=input_mask)
            
            sequence_output = self.output_dropout(sequence_output)
            
            if hasattr(self, "pooler"):
                pooled = self.pooler(sequence_output, mask=input_mask)
            else:
                pooled = self.output_dropout(pooled)
            
            encoded[head] = {
                "seq": sequence_output,
                "cls": cls_output,
                "pooled": pooled,
                "x": x_output
            }
        
        return encoded
    
    def forward_common(self, inputs, tag=True):
        encoded = self.joint_encode(inputs)
        aux_inputs = list()
        
        for head, sub_inputs in inputs.items():
            aux_inputs.append(
                (
                    sub_inputs.get("input_mask", None),
                    encoded[head]["seq"],
                    encoded[head]["pooled"],
                    sub_inputs.get("type_ids", None),
                    sub_inputs.get("shared_label_ids", None),
                    sub_inputs.get("all_outside_id", None),
                    sub_inputs.get("lang_id", None),
                    sub_inputs.get("domain_id", None)
                )
            )
        
        (
            input_mask, sequence_output, pooled, type_ids, 
            shared_label_ids, all_outside_id, lang_id, domain_id
        ) = zip(
            *aux_inputs
        )
        
        aux_inputs = dict(
            sequence_output=torch.cat(sequence_output),
            pooled=torch.cat(pooled),
            input_mask=None if input_mask[0] is None else torch.cat(input_mask),
            type_ids=None if type_ids[0] is None else torch.cat(type_ids),
            shared_label_ids=None if shared_label_ids[0] is None else torch.cat(shared_label_ids),
            all_outside_id=None if all_outside_id[0] is None else torch.cat(all_outside_id),
            lang_id=None if lang_id[0] is None else torch.cat(lang_id),
            domain_id=None if domain_id[0] is None else torch.cat(domain_id)
        )
        
        return encoded, aux_inputs
    
    def forward(self, inputs, tag=True, is_lm=False):
        if not is_lm:
            encoded, aux_inputs = self.forward_common(inputs)
            
            outputs = dict()
            aux_outputs = dict()
            
            if tag:
                for head, sub_inputs in inputs.items():
                    if head == "shared" or head == "unk":
                        continue
                    input_mask = sub_inputs.get("input_mask", None)
                    labels = sub_inputs.get("label_ids", None)
                    outputs[head] = dict(
                        ner=self[head](encoded[head]["seq"], mask=input_mask, labels=labels),
                        encoder=encoded[head]
                    )
            
            if self.config.all_shared and tag:
                aux_outputs["shared"] = self["shared"](
                    aux_inputs["sequence_output"],
                    mask=aux_inputs["input_mask"],
                    labels=aux_inputs["shared_label_ids"]
                )
            
            pooled = aux_inputs["pooled"]
            
            # language identification
            try:
                lang_clf = self["lang_clf"]
                aux_outputs["lang_clf"] = lang_clf(pooled, aux_inputs["lang_id"])
            except:
                pass
            
            # domain classification
            try:
                domain_clf = self["domain_clf"]
                aux_outputs["domain_clf"] = domain_clf(pooled, aux_inputs["domain_id"])
            except:
                pass
            
            # all O classification
            try:
                all_outside_clf = self["all_outside_clf"]
                aux_outputs["all_outside_clf"] = all_outside_clf(pooled, aux_inputs["all_outside_id"])
            except:
                pass
            
            return outputs, aux_outputs
        
        else:
            return self.forward_lm(inputs)


class XForMultiTokenClassificationBase(MultiTokenClassificationUtils):
    
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
        
        self.add_classifiers()
        self.add_lang_classifier()
        self.add_domain_classifier()
        self.add_all_outside_classifier()
        self.add_language_modeling()


class BertForMultiTokenClassification(bert.PreTrainedModel, XForMultiTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


class XLMRobertaForMultiTokenClassification(bert.PreTrainedModel, XForMultiTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


class XLMForMultiTokenClassification(xlm.PreTrainedModel, XForMultiTokenClassificationBase):
    
    def __init__(self, config):
        super().__init__(config)
        self.setup(config)
        self.init_weights()


MODEL_FOR_MULTI_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (xlm.Config, XLMForMultiTokenClassification),
        (xlmr.Config, XLMRobertaForMultiTokenClassification),
        (bert.Config, BertForMultiTokenClassification)
    ]
)


class AutoModelForMultiTokenClassification:
    
    def __init__(self):
        raise EnvironmentError(
            "AutoModelForMultiTokenClassification is designed to be instantiated "
            "using the `AutoModelForMultiTokenClassification.from_pretrained(pret"
            "rained_model_name_or_path)`."
        )
    
    @classmethod
    def get_models_map(cls, external_map=None):
        models_map = OrderedDict(MODEL_FOR_MULTI_TOKEN_CLASSIFICATION_MAPPING)
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
                            if k.startswith("classifiers") or k.startswith("aux_modules"):
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
