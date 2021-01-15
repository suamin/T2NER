# -*- coding: utf-8 -*-

import sys
import xml.etree.ElementTree as ET
import nltk
import re
import os
import random

seed = 42
random.seed(seed)

fixes = {
    "PHI-DATE": "DATE", 
    "PHI-DOCTOR": "NAME-DOCTOR", 
    "PHI-PATIENT": "NAME-PATIENT"
}

MASK_TOKEN = "____MASK____"


def read_i2b2_xml(fname, fine_grained=True, fix=False):
    root = ET.parse(fname).getroot()
    doc = root.find("TEXT").text
    tags = list(root.find("TAGS").iter())[1:] # first is "tags" element itself
    labels = list()
    for element in tags:
        labels.append((element.tag, element.attrib))
    
    masked_doc = ""
    masked = list()
    i = 0
    for tag, attr in labels:
        j, k = int(attr["start"]), int(attr["end"])
        chunk = doc[i:j]
        phi = doc[j:k]
        phi = re.sub(r"\s+", " ", phi) # normalize whitespace / newlines in entity
        if fine_grained:
            entity_type = tag + "-" + attr["TYPE"] if tag != attr["TYPE"] else tag
        else:
            et = attr["TYPE"]
            if tag == "PHI" and (et == "DOCTOR" or et == "PATIENT"):
                tag = "NAME"
            elif tag == "PHI" and et == "DATE":
                tag = "DATE"
            entity_type = tag
        if entity_type in fixes and fix:
            entity_type = fixes[entity_type]
        masked.append((phi, entity_type))
        masked_doc += chunk + " {}".format(MASK_TOKEN)
        i = k
    
    # heuristic to obtain good sentence tokenization
    masked_doc = re.sub(r"\n", " ", masked_doc)
    data = list()
    
    for sent in nltk.sent_tokenize(masked_doc):
        sent = re.sub(r"\s+", " ", sent.strip()).strip()
        if sent:
            n = len(re.findall(MASK_TOKEN, sent))
            data.append((sent, masked[:n]))
            masked = masked[n:]
    
    return data


def data_to_bio(data, ignore_meddocan_other=False):
    bio_corpus = list()
    for sent, labels in data:
        if labels:
            tokens = list()
            tags = list()
            i = 0
            for idx, token in enumerate(nltk.word_tokenize(sent)):
                if token == MASK_TOKEN:
                    temp = [""]
                else:
                    temp = token.split(MASK_TOKEN)
                for subtoken in temp:
                    if subtoken == "":
                        entity, entity_type = labels[i]
                        for j, entity_token in enumerate(nltk.word_tokenize(entity)):
                            tokens.append(entity_token)
                            if j == 0:
                                tags.append("B-" + entity_type)
                            else:
                                tags.append("I-" + entity_type)
                        i += 1
                    else:
                        tokens.append(subtoken)
                        tags.append("O")
        else:
            tokens = nltk.word_tokenize(sent)
            tags = ["O"] * len(tokens)
        bio = []
        for token, tag in zip(tokens, tags):
            if ignore_meddocan_other:
                if tag == "B-OTHER" or tag == "I-OTHER":
                    tag = "O"
            bio.append(token + " " + tag )
        bio_corpus.append(bio)
    return bio_corpus


def write_output(bio_corpus, fname):
    with open(fname, "w", encoding="utf-8", errors="ignore") as wf:
        temp = list()
        for sent in bio_corpus:
            temp.append("\n".join(sent))
        wf.write("\n\n".join(temp))


def convert_i2b2_deid_to_conll(fine_grained=True, fix=False):
    data_dir = "/raid/saam01/uned/data/sources/i2b2_deid"
    output_dir = "/raid/saam01/uned/data/sources/i2b2_conll"
    os.makedirs(output_dir, exist_ok=True)
    if fine_grained:
        output_dir = os.path.join(output_dir, "i2b2deid-fine")
    else:
        output_dir = os.path.join(output_dir, "i2b2deid")
    if fix:
        output_dir = os.path.join(output_dir + "-fix")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ("train", "test"):
        path = os.path.join(data_dir, split)
        bio_corpus = list()
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            bio_corpus.append(data_to_bio(read_i2b2_xml(file_path, fine_grained, fix)))
        if split == "train":
            random.shuffle(bio_corpus)
            # 790 total, take 700 for training, 90 for validation
            train_corpus = list()
            for doc in bio_corpus[:700]:
                train_corpus.extend(doc)
            write_output(train_corpus, os.path.join(output_dir, "train.txt"))
            
            valid_corpus = list()
            for doc in bio_corpus[700:]:
                valid_corpus.extend(doc)
            write_output(valid_corpus, os.path.join(output_dir, "dev.txt"))
        else:
            test_corpus = list()
            for doc in bio_corpus:
                test_corpus.extend(doc) 
            write_output(test_corpus, os.path.join(output_dir, "test.txt"))


def convert_meddocan_deid_to_conll(fine_grained=True, ignore_meddocan_other=False):
    data_dir = "/raid/saam01/uned/data/sources/MEDDOCAN"
    output_dir = "/raid/saam01/uned/data/sources/meddocan_conll"
    os.makedirs(output_dir, exist_ok=True)
    if fine_grained:
        output_dir = os.path.join(output_dir, "meddocandeid-fine")
    else:
        output_dir = os.path.join(output_dir, "meddocandeid")
    if ignore_meddocan_other:
        output_dir = os.path.join(output_dir + "-ignore")
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ("train", "dev", "test"):
        path = os.path.join(data_dir, split, "xml")
        bio_corpus = list()
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            bio_corpus.append(data_to_bio(read_i2b2_xml(file_path, fine_grained), ignore_meddocan_other))
        split_data = list()
        for doc in bio_corpus:
            split_data.extend(doc)
        write_output(split_data, os.path.join(output_dir, "{}.txt".format(split)))


if __name__=="__main__":
    dataset, fine_grained, fix_or_ignore = sys.argv[1:]
    if int(fine_grained) == 0:
        fine_grained = False
    else:
        fine_grained = True
    if int(fix_or_ignore)  == 0:
        fix_or_ignore = False
    else:
        fix_or_ignore = True
    if dataset == "i2b2":
        convert_i2b2_deid_to_conll(fine_grained, fix_or_ignore)
    else:
        convert_meddocan_deid_to_conll(fine_grained, fix_or_ignore)
