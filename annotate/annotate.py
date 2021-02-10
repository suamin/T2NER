# -*- coding: utf-8 -*-

import os
import logging
import pickle
import json
import argparse
import hashlib
import time

logger = logging.getLogger(__name__)

# CHANGE THIS LINE FOR YOUR OWN ENTITY TYPES
# TODO Add support for passing this via command line as a file
TYPES = ["PER", "ORG", "LOC", "OTH"]
LABELS = {idx: l for idx, l in enumerate(sorted({TYPES}))}


def init_logging(log_file):
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logging.basicConfig(
        handlers = [logging.StreamHandler()],
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO
    )
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S'
    ))
    return fh


def pretty_str(tokens, ids, labels, print_width=60):
    lens = [max(len(tokens[i]), len(ids[i]) + len(labels[i]) + 7) for i in range(len(tokens))]
    lines = list()
    tline = list()
    iline = list()
    rlen = 0
    tlen = sum(lens)
    if tlen <= print_width:
        print_width = tlen
    for idx, l in enumerate(lens):
        if rlen >= print_width:
            lines.append("    ".join([t.center(lj) for t, lj in tline]))
            lines.append("    ".join([i.center(lj) for i, lj in iline]))
            lines.append("\n")
            tline = list()
            iline = list()
            rlen = 0
        tline.append((tokens[idx], l))
        iline.append((ids[idx] + " -> " + labels[idx], l))
        rlen += l
    
    lines.append("    ".join([t.center(lj) for t, lj in tline]))
    lines.append("    ".join([i.center(lj) for i, lj in iline]))
    label_line = ">> labels = " + " ".join(["({} -> {})".format(k, v) for k, v in LABELS.items()])
    lines = [">> sentence = " + " ".join(tokens) + "\n", label_line + "\n", ">> tagged =\n"] + lines
    
    return "\n".join(lines)


class SentAnnObj:
    
    def __init__(self, tokens, label_map, print_width=60):
        assert len(tokens) > 0, "Empty sentence passed for annotation."
        self.tokens = tokens
        self.label_map = label_map
        self.labels = ["O"] * len(tokens)
        self.print_width = print_width
        self.hash = hashlib.md5(bytes(" ".join(tokens), "utf-8")).hexdigest()
    
    def __eq__(self, other):
        return self.hash == other.hash
    
    def add(self, start, label, end=None):
        assert 0 <= start < len(self.tokens), "Invalid start position"
        if end is None:
            end = start
        else:
            assert start < end < len(self.tokens), "Invalid end position" 
        assert label in self.label_map, (
            "Unknown label `{}`, expected index from map `{}`".format(label, self.label_map)
        )
        tag = self.label_map[label]
        for i in range(start, end + 1):
            if i == start:
                self.labels[i] = "B-{}".format(tag)
            else:
                self.labels[i] = "I-{}".format(tag)
    
    def clear(self):
        self.labels = ["O"] * len(self.tokens)
    
    @property
    def is_all_O(self):
        return sum([l == "O" for l in self.labels]) == len(self.labels)
    
    def __str__(self):
        ids = [str(i) for i in range(len(self.tokens))]
        return pretty_str(self.tokens, ids, self.labels, self.print_width)


class Annotater:
    
    @staticmethod
    def read_source(src_file):
        sentences = list()
        with open(src_file, encoding="utf-8", errors="ignore") as rf:
            tokens = list()
            for line in rf:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        sentences.append(tokens)
                        tokens = list()
                else:
                    splits = line.split("\t")
                    tokens.append(splits[0].strip())
            if tokens:
                sentences.append(tokens)
        return sentences
    
    @staticmethod
    def read_json_source(src_file):
        with open(src_file, encoding="utf-8", errors="ignore") as rf:
            idx2sents = json.load(rf)
        idx2sents = {int(idx): s for idx, s in idx2sents.items()}
        return idx2sents
    
    def stats(self, anns, total):
        return dict(
            annotated=sum([1 for idx, val in anns.items() if val[1] == 1]),
            skipped=sum([1 for idx, val in anns.items() if val[1] == 0]),
            revised=sum([1 for idx, val in anns.items() if val[1] == 2]),
            remaining=total - len(anns)
        )
    
    def log_progress(self, stats):
        A = stats["annotated"]
        Z = stats["revised"]
        S = stats["skipped"]
        R = stats["remaining"]
        logger.info("Progress ({:.3f}%): Annotated={} / Skipped={} / Remaining={} / Revised={}".format(
            (R/(A+S+R)) * 100, A, S, R, Z)
        )
    
    def init_session(self, args):
        # Raw tokenized sentences
        sentences = self.read_source(args.src_file)
        if args.translated_src_file:
            self.idx2sentence_en = self.read_json_source(args.translated_src_file)
        # Make working dir unique per source file
        work_dir = os.path.join(args.work_dir, os.path.splitext(os.path.split(args.src_file)[1])[0])
        resume = False
        if os.path.exists(work_dir):
            if not args.overwrite:
                raise Exception(
                    "Working directory `{}` exists already. Set `overwrite` flag to reset.".format(work_dir)
                )
            logger.info("Using existing directory `{}` as working directory.".format(work_dir))
            resume = True
        else:
            logger.info("Creating new working directory `{}`.".format(work_dir))
            os.makedirs(work_dir)
        
        log_file = os.path.join(work_dir, args.log_file)
        fh = init_logging(log_file)
        logger.addHandler(fh)
        
        progress_file = os.path.join(work_dir, "progress.json")
        ann_objs_file = os.path.join(work_dir, "anns.pkl")
        progress = dict(annotated=0, skipped=0, remaining=len(sentences))
        anns = dict()
        
        if resume:
            try:
                logger.info("Loading progress from : `{}` ...".format(progress_file))
                with open(progress_file) as rf:
                    progress = json.load(rf)
                logger.info("Loading annotations from : `{}` ...".format(ann_objs_file))
                with open(ann_objs_file, "rb") as rf:
                    anns = pickle.load(rf)
            except:
                pass    
        
        logger.info("Initializing annotation ...")
        logger.info("******   Annotation stats   ******")
        for k, v in progress.items():
            logger.info(" {} : {}".format(k, v))
        
        def get_confid(confid=None):
            if args.ask_for_confidence:
                while True:
                    y = input(">> INPUT : Confidence : ")
                    logger.info("AnnInput: `{}`".format(y))
                    try:
                        y = int(y)
                    except:
                        logger.error("Please provide rating as int in [1 - 5]")
                        continue
                    else:
                        if y in {1, 2, 3, 4, 5}:
                            confid = y
                            break
                        else:
                            logger.error("Please provide rating as in [1, 2, 3, 4, 5]")
                            continue
            else:
                if confid is None:
                    confid = 5
            return confid
        
        try:
            skipped_idxs = set()
            revised_idxs = set()
            revision_idxs = set()
            
            for idx, sent_tokens in enumerate(sentences):
                checked = idx in anns
                if checked:
                    ann_obj, status, confid = anns[idx]
                    if (status == 0 and args.review_skipped) or \
                       (status == 1 and args.revise_annotated) or \
                       (status == 2 and args.revise_annotated):
                        if args.filter_by_confidence > -1:
                            if args.filter_by_confidence in {1, 2, 3, 4, 5}:
                                if confid != args.filter_by_confidence:
                                    continue
                            else:
                                raise ValueError(
                                    "Invalid value for `filter_by_confidence`, should be in "
                                    "range [1, 5]."
                                )
                        if args.filter_by_status > -1:
                            if args.filter_by_status in {0, 1, 2}:
                                if status != args.filter_by_status:
                                    continue
                            else:
                                raise ValueError(
                                    "Invalid value for `filter_by_status`, should be one of "
                                    "0, 1 or 2."
                                )
                        if args.filter_by_token:
                            if args.filter_by_token.lower() not in \
                               set(" ".join(sent_tokens).lower().split(" ")):
                               continue
                        if args.filter_all_O:
                            if ann_obj.is_all_O:
                                continue
                        if args.filter_all_non_O:
                            if not ann_obj.is_all_O:
                                continue
                        if status == 0:
                            skipped_idxs.add(idx)
                        if status == 1:
                            revision_idxs.add(idx)
                        if status == 2:
                            revised_idxs.add(idx)
                    else:
                        continue
                else:
                    if args.review_skipped or args.revise_annotated:
                        continue
                    # Check if the sentence is duplicate
                    ann_obj = SentAnnObj(sent_tokens, LABELS)
                    has_dupe = False
                    for _, (ann_obj_i, _, confid) in anns.items():
                        if ann_obj == ann_obj_i:
                            logger.info("Found duplicate sentence `{}` with existing "
                                        "annotation, copying!".format(" ".join(ann_obj.tokens)))
                            ann_obj.labels = ann_obj_i.labels # Shallow copy will influence all refs
                            status = 1
                            has_dupe = True
                            anns[idx] = (ann_obj, status, confid)
                            self.log_progress(self.stats(anns, len(sentences)))
                            break
                    if not has_dupe:
                        try:
                            status = self.annotate(ann_obj, idx)
                        except Exception as e:
                            logger.error("Unknown exception : `{}` -- continuing".format(e))
                            continue
                        else:
                            confid = get_confid()
                            anns[idx] = (ann_obj, status, confid)
                            self.log_progress(self.stats(anns, len(sentences)))
            
            def review_or_revise(idxs, revised_idxs=None):
                j = 0
                idxs = sorted(list(set(idxs)))
                if revised_idxs is None:
                    revised_idxs = set()
                idxs_done = set()
                while True:
                    if len(idxs) == 0:
                        break
                    idxs = sorted(list(set(idxs)))
                    total = len(idxs)
                    remain = len(set(idxs) - set(idxs_done))
                    raw_progress = ((total - remain) / total) * 100
                    actual_progress = ((total - (total - len(idxs_done))) / total) * 100
                    logger.info(
                        "Raw Progress ({:.3f}%), Actual Progress ({:.3f}%): "
                        "Total = {}, Remain = {}, Revised/Reviewed={}, Annotated = {}".format(
                            raw_progress, actual_progress, total, remain, 
                            total - remain, len(idxs_done)
                        )
                    )
                    try:
                        if args.ask_ids_for_review_revise:
                            i = input("Enter sentence index to review = ")
                            i = int(i)
                            if args.revise_annotated:
                                if i in revised_idxs:
                                    revised_idxs.remove(i)
                                    if i in idxs_done:
                                        idxs_done.remove(i)
                                    idxs.append(i)
                            elif args.review_skipped:
                                if i in idxs_done:
                                    idxs_done.remove(i)
                                    idxs.append(i)
                            if i not in set(idxs):
                                logger.warning("Couldn't find this index in the set.")
                                continue
                        else:
                            i = None
                        if i is None:
                            if j == len(idxs):
                                break
                            idx = idxs[j]
                        else:
                            idx = i
                        ann_obj, old_status, confid = anns[idx]
                        status = self.annotate(ann_obj, idx, is_revision=args.revise_annotated)
                    except Exception as e:
                        logger.error("Unknown exception : `{}` -- continuing".format(e))
                        continue
                    else:
                        if (old_status == 2 or old_status == 1) and status == 0:
                            logger.warn("Tried to change revised / annotated sentence to skipped one.")
                            bad_overwrite = input("Are you sure (y/n)? ")
                            if bad_overwrite != "y":
                                continue
                        confid = get_confid(confid)
                        anns[idx] = (ann_obj, status, confid)
                        self.log_progress(self.stats(anns, len(sentences)))
                        if not args.ask_ids_for_review_revise:
                            j += 1
                        else:
                            if status == 2 and args.revise_annotated:
                                idxs.remove(i)
                            elif status == 1 and args.review_skipped:
                                idxs.remove(i)
                            if len(idxs) == 0:
                                break
                        
                        if status == 2 and args.revise_annotated:
                            idxs_done.add(idx)
                            revised_idxs.add(idx)
                        elif status == 1 and args.review_skipped:
                            idxs_done.add(idx)
            
            if args.review_skipped and skipped_idxs:
                logger.info("Reviewing skipped ...")
                review_or_revise(skipped_idxs)
            
            if not revision_idxs and revised_idxs:
                revision_idxs = set(revised_idxs)

            if args.revise_annotated and revision_idxs:
                logger.info("Revising annotated ...")
                review_or_revise(revision_idxs, revised_idxs=revised_idxs)
        
        except KeyboardInterrupt:
            logger.error("Early exiting!")
        
        with open(progress_file, "w") as wf:
            logger.info("Saving progrees at : `{}` ...".format(progress_file))
            progress = self.stats(anns, len(sentences)) 
            json.dump(progress, wf)
        
        with open(ann_objs_file, "wb") as wf:
            logger.info("Saving binary annotations at : `{}` ...".format(ann_objs_file))
            pickle.dump(anns, wf)
         
        json_anns_file = os.path.join(work_dir, "annotated.json")
        with open(json_anns_file, "w", encoding="utf-8", errors="ignore") as wf:
            logger.info("Saving (friendly) json annotations at : `{}` ...".format(json_anns_file))
            json_data = dict()
            for idx, (ann_obj_i, status, confid) in anns.items():
                json_data[idx] = {
                    "tokens": ann_obj_i.tokens, 
                    "labels": ann_obj_i.labels, 
                    "status": status,
                    "confid": confid
                }
            json.dump(json_data, wf, indent=1)
        
        friendly_anns_file = os.path.join(work_dir, "annotated_friendly.txt")
        with open(friendly_anns_file, "w", encoding="utf-8", errors="ignore") as wf:
            logger.info("Saving (more friendly) text annotations at : `{}` ...".format(friendly_anns_file))
            for idx, (ann_obj_i, status, confid) in anns.items():
                wf.write("STATUS={} | ID={} | CONFID={}\n".format(status, idx, confid))
                wf.write("=============================\n\n".format(status))
                for w, l in list(zip(ann_obj_i.tokens, ann_obj_i.labels)):
                    wf.write("{}\t{}\n".format(w.center(15), l.center(15)))
                wf.write("\n\n")
    
    def annotate(self, ann_obj, idx=None, is_revision=False):
        while True:
            print("\n>> ID = {}\n\n".format(idx) + str(ann_obj))
            print()
            x = input(">> input = ")
            logger.info("AnnInput: `{}`".format(x))
            x = x.strip()
            if x == "s":
                return 0
            elif x == "d":
                print(">> review = ")
                logger.info("\n" + str(ann_obj))
                y = input(">> INPUT : Are you sure? (y/n) : ")
                logger.info("AnnInput: `{}`".format(y))
                if y == "y":
                    if is_revision:
                        return 2
                    else:
                        return 1
            elif x == "c":
                ann_obj.clear()
                logger.info("cleared!")
            elif x == "h":
                if hasattr(self, "idx2sentence_en"):
                    if idx in self.idx2sentence_en:
                        translation = self.idx2sentence_en[idx]
                        logger.info("(Hint) {}".format(translation))
                    else:
                        logger.info("No hint available!")
                else:
                    logger.info("No hint available!")
            else:
                for i in x.split():
                    try:
                        temp = [int(j) for j in i.split(",")]
                        if len(temp) == 2:
                            start, label = temp
                            end = None
                        elif len(temp) == 3:
                            start, end, label = temp
                        else:
                            raise Exception
                    except Exception as e:
                        logger.error("Invalid input `{}` : `{}` -- ignored!".format(i, e))
                    else:
                        try:
                            ann_obj.add(start, label, end)
                        except Exception as e:
                            logger.error("Couldn't add input `{}` : `{}`".format(i, e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--work_dir", 
        default=None, type=str, required=True, 
        help="The directory where the annotator progress will be saved."
    )
    parser.add_argument(
        "--src_file", 
        default=None, type=str, required=True,
        help="Input text file in CoNLL format to be annotated."
    )
    parser.add_argument(
        "--translated_src_file", 
        default=None, type=str,
        help="Translated EN texts for hints during annotation."
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true",
        help="Overwrite the progress in pre-existing directory."
    )
    parser.add_argument(
        "--review_skipped", 
        action="store_true",
        help="Review previously skipped examples for annotation."
    )
    parser.add_argument(
        "--revise_annotated", 
        action="store_true",
        help="Revise previously annotated examples."
    )
    parser.add_argument(
        "--ask_ids_for_review_revise",
        action="store_true",
        help="In review or revision mode, the tool will ask for the sentence "
             "index to select from the pool otherwise it will present "
             "sentences sequentially in order."
    )
    parser.add_argument(
        "--ask_for_confidence",
        action="store_true",
        help="After every annotation the annotator will be prompted to rate the "
             "annotation with confidence value in [1, 2, 3, 4, 5] with 5 being "
             "highest. Lower confidence can be interpreted as more ambiguous. "
             "When disabled, all annotations are given 5 by default."
    )
    parser.add_argument(
        "--filter_by_confidence",
        default=-1, type=int,
        help="If a value is provided in [1, 2, 3, 4, 5] then only those confidence "
             "valued annotations will be prompted to annotator during review or revise."
    )
    parser.add_argument(
        "--filter_by_status",
        default=-1, type=int,
        help="If a value is provided in [0, 1, 2] then only those status "
             "valued annotations will be prompted to annotator during review or revise."
    )
    parser.add_argument(
        "--filter_by_token",
        default=None, type=str,
        help="If a value is provided in then only those sentences containing this "
             "token will be prompted to annotator during review or revise."
    )
    parser.add_argument(
        "--filter_all_O",
        action="store_true",
        help="If this flag is set then all sentences where all tokens are labeled as "
             "O will be ignored to be prompted to annotator during review or revise."
    )
    parser.add_argument(
        "--filter_all_non_O",
        action="store_true",
        help="If this flag is set then all sentences where all tokens are labeled as "
             "O will only be prompted to annotator during review or revise."
    )
    parser.add_argument(
        "--log_file", 
        default="log.txt", type=str,
        help="Log file name (not path!, this will be created under output directory)."
    )
    args = parser.parse_args()
    ann = Annotater()
    ann.init_session(args)
