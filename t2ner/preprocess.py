# coding=utf-8
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script adapted from: https://github.com/google-research/xtreme/blob/master/utils_preprocess.py

from __future__ import absolute_import, division, print_function

import os
import argparse
from transformers import AutoTokenizer, XLMRobertaTokenizer


def preprocess_one_file(infile, outfile, tokenizer, max_len, wikiann=False, wnut=False):
    
    if not os.path.exists(infile):
        print("{} does not exist".format(infile))
        return 0, set()
    
    special_tokens_count = 3 if isinstance(tokenizer, XLMRobertaTokenizer) else 2
    max_seq_len = max_len - special_tokens_count
    subword_len_counter = 0
    labels = set()
    
    with open(infile, "r", encoding="utf-8", errors="ignore") as fin, \
         open(outfile, "w", encoding="utf-8", errors="ignore") as fout:
         
         for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                subword_len_counter = 0
                continue
            
            items = line.split()
            token = items[0].strip()
            
            if wikiann:
                token = token.split(":")[1] # first is language code
                if token == "": # : token
                    token = ":"
            
            if len(items) > 1:
                label = items[-1].strip()
                if wnut:
                    label = label.upper()
            else:
                label = "O"
            
            labels.add(label)
            
            subword_tokens = tokenizer.tokenize(token)
            if not subword_tokens:
                continue
            current_subwords_len = len(subword_tokens)
            
            if (subword_len_counter + current_subwords_len) >= max_seq_len:
                fout.write("\n")
                fout.write("{}\t{}\n".format(token, label))
                subword_len_counter = current_subwords_len
            else:
                fout.write("{}\t{}\n".format(token, label))
                subword_len_counter += current_subwords_len
    
    return 1, labels


def preprocess(args):
    model_type = args.model_type
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None
    )
    output_dir = os.path.join(args.output_dir, args.model_name_or_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for lang in args.languages.split(","):
        lang_dir = os.path.join(args.data_dir, lang)
        # dataset: domain_datasetname
        for dataset in os.listdir(lang_dir):
            domain, dataset_name = dataset.split("_")
            wikiann = dataset_name == "wikiann"
            wnut = dataset_name == "wnut17"
            final_dir = os.path.join(lang_dir, dataset)
            labels = set()
            for file in os.listdir(final_dir):
                fname, ext = os.path.splitext(file)
                if fname == "extra": # sometimes appear in wikiann
                    continue
                if fname == "valid" or fname == "devel":
                    split = "dev"
                else:
                    split = fname
                
                infile = os.path.join(final_dir, file)
                outfile = os.path.join(output_dir, "{}.{}.{}-{}".format(lang, domain, dataset_name, split))
                if os.path.exists(outfile) and not args.overwrite_output_dir:
                    continue
                
                code, file_labels = preprocess_one_file(
                    infile, outfile, tokenizer, args.max_len, wikiann, wnut
                )
                labels.update(file_labels)
                if code > 0:
                    print("Finished preprocessing {}".format(outfile))
            
            labels_file = os.path.join(output_dir, "{}.{}.{}.labels".format(lang, domain, dataset_name))
            if os.path.exists(labels_file) and not args.overwrite_output_dir:
                continue
            with open(labels_file, "w") as wf:
                for label in sorted(labels):
                    wf.write("{}\n".format(label))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                      help="The `ner` data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                      help="The output data dir where any processed files will be written to.")
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str,
                      help="The pre-trained model.")
    parser.add_argument("--model_type", default="bert", type=str,
                      help="Model type of tokenizer.")
    parser.add_argument("--max_len", default=128, type=int,
                      help="Maximum length of sentences.")
    parser.add_argument("--do_lower_case", action='store_true',
                      help="Whether to do lower case.")
    parser.add_argument("--cache_dir", default=None, type=str,
                      help="Cache directory.")
    parser.add_argument("--overwrite_output_dir", action='store_true',
                      help="Overwrite existing output files.")
    parser.add_argument("--languages", default="en", type=str,
                      help="Comma separated names of languages to preprocess.")
    args = parser.parse_args()
    preprocess(args)
