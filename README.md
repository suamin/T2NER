# T2NER

A transformers based transfer learning framework for named entity recognition (NER).

## Requirements
Please install the HuggingFace [trasnformers](https://github.com/huggingface/transformers) library.

## Preprocessing
First preprocess the CoNLL format data:

```
python t2ner/preprocess.py \
    --data_dir data/ner \
    --output_dir data/processed \
    --model_name_or_path bert-base-multilingual-cased \
    --model_type bert \
    --max_len 128 \
    --overwrite_output_dir \
    --languages en,es,nl
```

## Experiments
To run an experiment:

```
python t2ner/run.py \
    --exp_type ner \
    --base_json configs/base.json \
    --exp_json configs/ner.json

```

## TODOs
- Add documentation
- Update README