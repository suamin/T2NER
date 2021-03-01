# -*- coding: utf-8 -*-

import sys
import os
import dataclasses
import json
import argparse

from pathlib import Path
from transformers import HfArgumentParser, TrainingArguments

from t2ner.trainers import NERTrainer, NERArguments
from t2ner.trainers import MultiTaskNERTrainer, MultiTaskNERArguments
from t2ner.trainers import GRLTrainer, GRLArguments
from t2ner.trainers import EMDTrainer, EMDArguments
from t2ner.trainers import KeungTrainer, KeungArguments
from t2ner.trainers import MCDTrainer, MCDArguments
from t2ner.trainers import MMETrainer, MMEArguments
from t2ner.trainers import EntMinTrainer, EntMinArguments

from t2ner.models import ModelArguments
from t2ner.models import AutoModelForTokenClassification
from t2ner.models import AutoModelForMultiTokenClassification


def main(args):
    base_json = args.base_json
    exp_json = args.exp_json
    
    if args.exp_type == "ner":
        parser = HfArgumentParser((TrainingArguments, ModelArguments, NERArguments))
        trainer = NERTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        trainer.run(AutoModelForTokenClassification)
    
    elif args.exp_type == "unsup_adapt":
        if args.method == "grl":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, GRLArguments))
            trainer = GRLTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        elif args.method == "emd":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, EMDArguments))
            trainer = EMDTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        elif args.method == "keung":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, KeungArguments))
            trainer = KeungTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        elif args.method == "mcd":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, MCDArguments))
            trainer = MCDTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        elif args.method == "mme":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, MMEArguments))
            trainer = MMETrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        else:
            raise NotImplementedError("Unknown method.")
        trainer.run(AutoModelForTokenClassification)
    
    elif args.exp_type == "ssl":
        if args.method == "entmin":
            parser = HfArgumentParser((TrainingArguments, ModelArguments, EntMinArguments))
            trainer = EntMinTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        else:
            raise NotImplementedError("Unknown method.")
        trainer.run(AutoModelForTokenClassification)
    
    elif args.exp_type == "multitask":
        parser = HfArgumentParser((TrainingArguments, ModelArguments, MultiTaskNERArguments))
        trainer = MultiTaskNERTrainer(*parse_json_config(parser.dataclass_types, base_json, exp_json))
        trainer.run(AutoModelForMultiTokenClassification)
    
    else:
        raise NotImplementedError("Implment please!")


def parse_json_config(dataclass_types, base_json, exp_json=None):
    data = json.loads(Path(base_json).read_text())
    if exp_json:
        exp_data = json.loads(Path(exp_json).read_text())
        for k, v in exp_data.items():
            data[k] = v
    outputs = []
    for dtype in dataclass_types:
        keys = {f.name for f in dataclasses.fields(dtype)}
        inputs = {k: v for k, v in data.items() if k in keys}
        obj = dtype(**inputs)
        outputs.append(obj)
    return (*outputs,)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument(
        "--exp_type",
        default=None, type=str, required=True,
        help="Type of experiment {ner, unsup_adapt, ...}."
    )
    parser.add_argument(
        "--base_json", 
        default=None, type=str, required=True,
        help="Common JSON config file."
    )
    parser.add_argument(
        "--exp_json", 
        default=None, type=str, required=True,
        help="Experiment specific JSON config file."
    )
    parser.add_argument(
        "--method",
        default=None, type=str,
        help="Sub-method in a specific experiment."
    )
    args = parser.parse_args()
    main(args)
