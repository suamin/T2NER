# -*- coding: utf-8 -*-

from transformers import BertConfig, BertPreTrainedModel, BertModel
from transformers import XLMConfig, XLMPreTrainedModel, XLMModel
from transformers import XLMRobertaConfig, XLMRobertaModel

from collections import namedtuple

HFModel = namedtuple("HFModel", "Config PreTrainedModel Model")

bert = HFModel(BertConfig, BertPreTrainedModel, BertModel)
xlm = HFModel(XLMConfig, XLMPreTrainedModel, XLMModel)
xlmr = HFModel(XLMRobertaConfig, BertPreTrainedModel, XLMRobertaModel)

HF_MODELS = [bert, xlm, xlmr]

from .token_classification import AutoModelForTokenClassification
from .multi_token_classification import AutoModelForMultiTokenClassification
from .base import ModelArguments