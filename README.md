# T2NER

A transformers based transfer learning framework for named entity recognition (NER).

#### Instructions

Clone the repository and run the requirements file:
```
git clone https://github.com/suamin/t2ner.git
cd t2ner
pip install -r requirements
```

#### Preprocessing

Download the NER data of interest and convert it into CoNLL format. Example datasets are provided in `data` folder (GermEval 2014, CoNLL-2002). Then, preprocess the CoNLL formatted data:

```
python t2ner/preprocess.py \
    --data_dir data/ner \
    --output_dir data/processed \
    --model_name_or_path bert-base-multilingual-cased \
    --model_type bert \
    --max_len 128 \
    --overwrite_output_dir \
    --languages es,nl
```

#### Experiments

To run an experiment:

```
python t2ner/run.py \
    --exp_type ner \
    --base_json configs/base.json \
    --exp_json configs/ner.json

```

## Citation

If you find our framework useful, please consider citing:

```bibtex
@inproceedings{amin-neumann-2021-t2ner,
    title = "{T}2{NER}: Transformers based Transfer Learning Framework for Named Entity Recognition",
    author = "Amin, Saadullah and Neumann, G{\"u}nter",
    booktitle = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations",
    month = apr,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.eacl-demos.25",
    doi = "10.18653/v1/2021.eacl-demos.25",
    pages = "212--220"
}
```

Also, check our follow-up work using `T2NER` for few-shot cross-lingual de-identification of clinical texts:

```bibtex
@inproceedings{amin-etal-2022-shot,
    title = "Few-Shot Cross-lingual Transfer for Coarse-grained De-identification of Code-Mixed Clinical Texts",
    author = "Amin, Saadullah and Pokaratsiri Goldstein, Noon and Kelly Wixted, Morgan and Garcia-Rudolph, Alejandro and Mart{\'\i}nez-Costa, Catalina and Neumann, G{\"u}nter",
    booktitle = "Proceedings of the 21st Workshop on Biomedical Language Processing",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.bionlp-1.20",
    doi = "10.18653/v1/2022.bionlp-1.20",
    pages = "200--211"
}
```

## Acknowledgements 

The algorithmic components of the framework largely follow [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library) and [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch), if you find `T2NER` useful, please also consider citing these works.
