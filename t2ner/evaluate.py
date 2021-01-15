# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import os
import logging

from seqeval.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

from tqdm import tqdm

from .base import TrainInferenceUtils
from .data import utils, IGNORE_INDEX

logger = logging.getLogger(__name__)


class Evaluator(TrainInferenceUtils):
    
    def __init__(self, **kwargs):
        self.per_device_eval_batch_size = kwargs.get("per_device_eval_batch_size", 32)
        self.no_cuda = kwargs.get("no_cuda", False)
        self.verbose = kwargs.get("verbose", True)
    
    def evaluate(
        self,
        model,
        dataloader,
        id2label=None,
        name="",
        predict_only=False,
        loss_only=False
    ):
        eval_batch_size = self.eval_batch_size
        logger.info("***** Running evaluation %s *****" % name)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Eval batch size = %d", eval_batch_size)
        
        eval_loss = 0.
        nb_eval_steps = 0
        preds = None
        has_crf = False
        out_label_ids = None
        has_labels = True if not predict_only else False
        model.eval()
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            if "input_ids" in batch:
                pred_layer = None
                for tensor_key in batch:
                    batch[tensor_key] = batch[tensor_key].to(self.device)
                batch_label_ids = batch.get("label_ids", None)
            else:
                # If multiple-NER setting (i.e. multiple classification layers 
                # for different datasets in multi-dataset setting).
                if len(batch) > 1:
                    raise ValueError(
                        "For evaluation with more than one output classification "
                        "layer, only one layer can be evaluated at a time."
                    )
                
                pred_layer = list(batch.keys())[0]
                for tensor_key in batch[pred_layer]:
                    batch[pred_layer][tensor_key] = (
                        batch[pred_layer][tensor_key].to(self.device)
                    )
                        
                if pred_layer == "shared":
                    batch_label_ids = batch[pred_layer].get("shared_label_ids", None)
                else:
                    batch_label_ids = batch[pred_layer].get("label_ids", None)
            
            # TODO Remove: normally it should not happen but if we have mixed 
            # batches with/without labels, then if there is any unlabeled batch, 
            # hard set has_labels to False
            if batch_label_ids is None:
                has_labels = False
            
            with torch.no_grad():
                outputs = model(batch)
                
                if pred_layer is None:
                    outputs = outputs["ner"]
                else:
                    # second index is aux_output; used when using shared labels
                    if pred_layer == "shared":
                        outputs = outputs[1][pred_layer]
                    else:
                        outputs = outputs[0]
                        outputs = outputs[pred_layer]["ner"]
                
                logits = outputs["logits"]
                
                # for CRF output layers, the decoded sequence is attached
                if "prediction" in outputs:
                    has_crf = True
                    crf_predictions = outputs["prediction"]
                
                if has_labels:
                    tmp_eval_loss = outputs["loss"]
                    if self.n_gpu > 1:
                        tmp_eval_loss = tmp_eval_loss.mean()
                    eval_loss += tmp_eval_loss
            
            nb_eval_steps += 1
            
            if has_crf:
                if preds is None:
                    preds = self.detach(crf_predictions)
                else:
                    preds = np.append(preds, self.detach(crf_predictions), axis=0)
            else:
                if preds is None:
                    preds = self.detach(logits)
                else:
                    preds = np.append(preds, self.detach(logits), axis=0)
            
            if has_labels:
                batch_label_ids = self.detach(batch_label_ids)
                if out_label_ids is None:
                    out_label_ids = batch_label_ids
                else:
                    out_label_ids = np.append(out_label_ids, batch_label_ids, axis=0)
        
        results = {"loss": 0.}
        preds_list = []
        
        if nb_eval_steps != 0 and has_labels:
            eval_loss /= nb_eval_steps
            results["loss"] = eval_loss.item()
            preds_list, out_label_list = self.align_predictions(
                preds, out_label_ids, id2label
            )
            
            if not loss_only:
                results["precision"] = precision_score(out_label_list, preds_list)
                results["recall"] = recall_score(out_label_list, preds_list)
                results["f1"] = f1_score(out_label_list, preds_list)
                results["report"] = "\n" + classification_report(
                    out_label_list, preds_list, digits=4
                )
                
                ntotal, correct = 0, list()
                for out_label, pred in zip(out_label_list, preds_list):
                    if sum(l == "O" for l in out_label) == len(out_label):
                        correct.append(sum(l == "O" for l in pred) == len(out_label))
                        ntotal += 1
                results["accuracy"]  = sum(correct) / ntotal
            
            if self.verbose:
                logger.info("***** Evaluation result *****")
                for key in sorted(results.keys()):
                    logger.info("  %s = %s", key, str(results[key]))
        
        else:
            preds_list = preds
        
        return results, preds_list
    
    @staticmethod
    def save_results(results, output_dir, name="", split="dev"):
        output_file = os.path.join(output_dir, "{}_{}_results.txt".format(split, name))
        with open(output_file, "w") as wf:
            for key, value in results.items():
                wf.write("%s = %s\n" % (key, value))
    
    @staticmethod
    def align_predictions(predictions, label_ids, id2label=None):
        assert predictions.shape[0] == label_ids.shape[0]
        if len(predictions.shape) == 3:
            preds = np.argmax(predictions, axis=2)
        else:
            preds = predictions
        num_examples, seq_len = label_ids.shape
        out_label_list = [[] for _ in range(num_examples)]
        preds_list = [[] for _ in range(num_examples)]
        
        for i in range(num_examples):
            for j in range(seq_len):
                if label_ids[i, j] != IGNORE_INDEX:
                    o = label_ids[i][j]
                    p = preds[i][j]
                    out_label_list[i].append(id2label[o] if id2label else o)
                    preds_list[i].append(id2label[p] if id2label else p)
        
        return preds_list, out_label_list 
    
    @staticmethod
    def save_predictions(preds_list, input_file, output_file):
        examples = utils.read_conll_ner_file(input_file)
        assert len(preds_list) == len(examples)
        
        with open(output_file, "w", encoding="utf-8", errors="ignore") as wf:
            example_id = 0
            for idx, example in enumerate(examples):
                words = example["words"]
                preds = preds_list[idx]
                assert len(words) == len(preds)
                for word, pred in zip(words, preds):
                    wf.write("{}\t{}\n".format(word, pred))
                wf.write("\n")
    
    def run_eval_on_collection(
        self,
        model,
        collection,
        output_dir,
        split="dev",
        input_dir=None, 
        loss_only=False,
        predict_only=False,
        save_predictions=False
    ):
        eval_results = dict()
        
        for unique_name, (dataloader, id2label) in collection.items():
            results, preds = self.evaluate(
                model=model,
                dataloader=dataloader,
                id2label=id2label,
                name=unique_name,
                predict_only=predict_only,
                loss_only=loss_only
            )
            eval_results[unique_name] = results
            self.save_results(results, output_dir, unique_name, split)
            
            if save_predictions:
                # TODO remove dependence on source file
                if input_dir is None:
                    raise ValueError(
                        "For saving predictions, input_dir is reequired "
                        "to read the source file."
                    )
                
                # Check if shared
                temp = unique_name.split(".")
                if temp[-1] == "shared":
                    input_fname = "{}-{}".format(".".join(temp[:-1]), split)
                    output_fname =  "predictions_{}_shared.txt".format(input_fname)
                else:
                    input_fname = "{}-{}".format(unique_name, split)
                    output_fname = "predictions_{}.txt".format(input_fname)
                
                input_file = os.path.join(input_dir, input_fname)
                output_file = os.path.join(output_dir, output_fname)
                self.save_predictions(preds, input_file, output_file)
        
        return eval_results
