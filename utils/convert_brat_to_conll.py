# -*- coding: utf-8 -*-

import os
import sys
import glob

brat_script = os.path.join(os.path.dirname(__file__), '../tmp/clones/brat/tools/anntoconll.py')
data = "meddocan"


def standarize_bratconll_to_conll(fname):
    lines = list()
    with open(fname, encoding="utf-8", errors="ignore") as rf:
        for line in rf:
            if not line.strip():
                lines.append(line)
            else:
                label, _, _, token = line.strip().split("\t")
                lines.append("{}\t{}\n".format(token, label))
    return lines


if data == "quaero":
    QUAERO_PATH = "data/quaero/QUAERO_FrenchMed/corpus/{}/MEDLINE"
    for split in ["train", "dev", "test"]:
        split_dir = QUAERO_PATH.format(split)
        for fname in os.listdir(split_dir):
            fname = os.path.join(split_dir, fname)
            os.system("python {} {}".format(brat_script, fname))
    
    # Standardize to CoNLL format
    QUAERO_PATH = "data/quaero/QUAERO_FrenchMed/corpus/{}/MEDLINE/*.conll"
    for split in ["train", "dev", "test"]:
        split_lines = list()
        for fname in glob.iglob(QUAERO_PATH.format(split)):
            split_lines.extend(standarize_bratconll_to_conll(fname) + ["\n"])
        
        output_fname = "data/quaero/QUAERO_FrenchMed/corpus/{}/MEDLINE/{}".format(split, split + ".conll")
        with open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
            for line in split_lines:
                _ = wf.write(line)

elif data == "cantemist":
    CANTEMIST_PATH = "data/cantemist/{}-to-publish/cantemist-ner"
    for split in ["train-set", "dev-set1"]:
        split_dir = CANTEMIST_PATH.format(split)
        for fname in os.listdir(split_dir):
            fname = os.path.join(split_dir, fname)
            os.system("python {} {}".format(brat_script, fname))
    
    # Standardize to CoNLL format
    CANTEMIST_PATH = "data/cantemist/{}-to-publish/cantemist-ner/*.conll"
    for split in ["train-set", "dev-set1"]:
        split_lines = list()
        for fname in glob.iglob(CANTEMIST_PATH.format(split)):
            split_lines.extend(standarize_bratconll_to_conll(fname) + ["\n"])
        
        output_fname = "data/cantemist/{}-to-publish/cantemist-ner/{}".format(split, split + ".conll")
        with open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
            for line in split_lines:
                _ = wf.write(line)

elif data == "meddocan":
    MEDDOCAN_PATH = "data/sources/MEDDOCAN/{}/brat"
    for split in ["train", "dev", "test"]:
        split_dir = MEDDOCAN_PATH.format(split)
        for fname in os.listdir(split_dir):
            fname = os.path.join(split_dir, fname)
            os.system("python {} {}".format(brat_script, fname))

    # Standardize to CoNLL format
    MEDDOCAN_PATH = "data/sources/MEDDOCAN/{}/brat/*.conll"
    for split in ["train", "dev", "test"]:
        split_lines = list()
        for fname in glob.iglob(MEDDOCAN_PATH.format(split)):
            split_lines.extend(standarize_bratconll_to_conll(fname) + ["\n"])
        
        output_fname = "data/sources/MEDDOCAN/{}/{}".format(split, split + ".conll")
        with open(output_fname, "w", encoding="utf-8", errors="ignore") as wf:
            for line in split_lines:
                _ = wf.write(line)
