# -*- coding: utf-8 -*-

import logging
import os
import sys
import random
import json
import time

from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from dataclasses import dataclass, field
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from .base import TrainInferenceUtils
from .evaluate import Evaluator, logger as eval_logger

from .optim import build_optimizer, build_lr_scheduler

logger = logging.getLogger(__name__)


class BaseTrainer(ABC, TrainInferenceUtils):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__()
        
        self.training_args = training_args
        self.model_args = model_args
        self.exp_args = exp_args
        self.seed = training_args.seed
        
        self.setup_output_dir()
        self.setup_logging()
        self.setup_config_kwargs()
        self.setup_evaluator()
        
        self._models = OrderedDict()
        self._optims = OrderedDict()
        self._scheds = OrderedDict()
        
        logger.info("Input args: %r %r %r", model_args, exp_args, training_args)
        logger.info("Process device: %s, n_gpu: %s", self.device, self.n_gpu)
    
    def setup_output_dir(self):
        # Check if output directory is empty
        if os.path.exists(self.training_args.output_dir) and \
           os.listdir(self.training_args.output_dir) and \
           self.training_args.do_train and \
           not self.training_args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
                .format(self.training_args.output_dir)
            )
        
        # Create output directory if needed
        if not os.path.exists(self.training_args.output_dir):
            os.makedirs(self.training_args.output_dir)
    
    def setup_logging(self):
        # Setup logging
        log_file = os.path.join(self.training_args.output_dir, "log.txt")
        fh = self.init_logging(log_file)
        logger.addHandler(fh)
        eval_logger.addHandler(fh)
    
    def setup_config_kwargs(self):
        self.__setattr__("config_kwargs", dict(
            mode=self.model_args.encoder_mode,
            use_crf=self.model_args.use_crf,
            crf_bigram=self.model_args.crf_bigram,
            normalize_clf=self.model_args.normalize_clf,
            temp_clf=self.model_args.temp_clf,
            output_classifier_dropout=self.model_args.output_classifier_dropout,
            loss_fct=self.model_args.loss_fct,
            ignore_heads=self.model_args.ignore_heads,
            add_xnet=self.model_args.add_xnet,
            class_num_list=self.model_args.class_num_list,
            valid_metric=self.model_args.valid_metric
        ))
    
    def setup_evaluator(self):
        self.__setattr__("evaluator",  Evaluator(
            per_device_eval_batch_size=self.training_args.per_device_eval_batch_size,
            no_cuda=self.training_args.no_cuda,
            verbose=True
        ))
    
    @property
    def features_kwargs(self):
        # Set features options
        tokenizer_name_or_path = self.model_args.model_name_or_path
        if self.model_args.tokenizer_name_or_path is not None:
            tokenizer_name_or_path = self.model_args.tokenizer_name_or_path
        features_kwargs = dict(
            tokenizer_name_or_path=tokenizer_name_or_path,
            do_lower_case=self.model_args.do_lower_case,
            max_seq_length=self.model_args.max_seq_length,
            cache_dir=self.model_args.cache_dir
        )
        return features_kwargs

    @property
    def data_kwargs(self):
        data_kwargs = dict(
            preprocessed_data_dir=self.exp_args.preprocessed_data_dir,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            features_cache_dir=self.model_args.features_cache_dir,
            overwrite_cache=self.model_args.overwrite_cache,
            features_kwargs=self.features_kwargs
        )
        return data_kwargs
    
    @property
    def is_max_step(self):
        if self.training_args.max_steps > 0 and \
           self.global_step > self.training_args.max_steps:
            return True
        else:
            return False
    
    @property
    def is_update_step(self):
        # Requires external attributes (step, num_batches)
        if (self.step + 1) % self.training_args.gradient_accumulation_steps == 0 or (
            # last step in epoch but step is always smaller than gradient_accumulation_steps
            self.num_batches <= self.training_args.gradient_accumulation_steps
            and (self.step + 1) == self.num_batches
        ):
            return True
        else:
            return False
    
    @property
    def progress(self):
        # Requires external attributes (epoch, step, num_batches)
        return self.epoch + ((self.step + 1) / self.num_batches)
    
    @property
    def total_train_batch_size(self):
        return (
            self.training_args.train_batch_size * 
            self.training_args.gradient_accumulation_steps
        )
    
    @property
    def train_dataloader_length(self):
        return self._train_dataloader_length
    
    @train_dataloader_length.setter
    def train_dataloader_length(self, value):
        self._train_dataloader_length = value
    
    @property
    def t_total(self):
        if self.training_args.max_steps > 0:
            t_total = self.training_args.max_steps
            num_train_epochs = (
                self.training_args.max_steps // 
                (self.train_dataloader_length // self.training_args.gradient_accumulation_steps) 
                + 1
            )
            self.training_args.num_train_epochs = num_train_epochs
        else:
            t_total = int(
                self.train_dataloader_length // 
                self.training_args.gradient_accumulation_steps * 
                self.training_args.num_train_epochs
            )
        return t_total
    
    def register_optim_and_scheduler(self, name="model", optim=None, sched=None):
        if self.__dict__.get("_optims") is None:
            raise AttributeError(
                "Cannot assign optim before super().__init__() call"
            )
        
        if self.__dict__.get("_scheds") is None:
            raise AttributeError(
                "Cannot assign sched before super().__init__() call"
            )
        
        self._optims[name] = optim
        self._scheds[name] = sched
    
    def register_model(self, model, name="model"):
        if self.__dict__.get("_models") is None:
            raise AttributeError(
                "Cannot assign model before super().__init__() call"
            )
        
        self._models[name] = model
    
    @property
    def model(self):
        return self._models["model"]
    
    @model.setter
    def model(self, model):
        self.register_model(model, name="model")
    
    def __getitem__(self, name):
        optim = self._optims[name]
        sched = self._scheds[name]
        return optim, sched
    
    def get_optimizer_and_scheduler(
        self,
        model,
        optim="adamw",
        optim_kwargs=dict(),
        lr_scheduler="constant",
        lr_scheduler_kwargs=dict()
    ):
        if "lr" not in optim_kwargs:
            optim_kwargs["lr"] = self.training_args.learning_rate
        if "weight_decay" not in optim_kwargs:
            optim_kwargs["weight_decay"] = self.training_args.weight_decay
        if optim == "adamw" and "adam_epsilon" not in optim_kwargs:
            optim_kwargs["adam_epsilon"] = self.training_args.adam_epsilon
        
        optimizer = build_optimizer(model, optim, **optim_kwargs)
        
        if lr_scheduler != "constant":
            try:
                t_total = self.t_total
            except:
                t_total = -1
            if "num_training_steps" not in lr_scheduler_kwargs:
                lr_scheduler_kwargs["num_training_steps"] = t_total
            if "warmup_steps" not in lr_scheduler_kwargs:
                lr_scheduler_kwargs["warmup_steps"] = self.training_args.warmup_steps
        
        lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler, **lr_scheduler_kwargs)
        
        return optimizer, lr_scheduler
    
    @property
    def model_optim_config(self):
        return self._model_optim_config
    
    @model_optim_config.setter
    def model_optim_config(self, optim_config):
        self._model_optim_config = optim_config
    
    @property
    def extended_optim_config_dict(self):
        return self._extended_optim_config_dict
    
    @extended_optim_config_dict.setter
    def extended_optim_config_dict(self, optim_config_dict):
        self._extended_optim_config_dict = optim_config_dict
    
    def configure_optimization(self):
        optim_config_dict = {"model": self.model_optim_config}
        if hasattr(self, "extended_optim_config_dict"):
            extended = self.extended_optim_config_dict
        else:
            extended = None
        if extended is not None:
            for name, item in extended.items():
                if name == "model":
                    raise ValueError(
                        "The name 'model' is reserved for optimization"
                    )
                optim_config_dict[name] = item
        
        for name, item in optim_config_dict.items():
            optim, lr_scheduler = self.get_optimizer_and_scheduler(
                item[name],
                optim=item["optim"],
                optim_kwargs=item.get("optim_kwargs", dict()),
                lr_scheduler=item["lr_scheduler"],
                lr_scheduler_kwargs=item.get("lr_scheduler_kwargs", dict())
            ) 
            self.register_optim_and_scheduler(name=name, optim=optim, sched=lr_scheduler)
    
    def get_last_lr(self, names=["model"]):
        last_lrs = dict()
        for name in names:
            if self._scheds[name] is not None:
                last_lrs[name] = self._scheds[name].get_last_lr()[0]
        return last_lrs
    
    def model_to_gpu(self, model):
        if self.n_gpu > 1:
            model = nn.DataParallel(model)
        model.to(self.device)
        return model
    
    def update_lr(self, names=None):
        if names is None:
            names = list(self._scheds.keys())
        for name in names:
            if self._scheds[name] is not None:
                self._scheds[name].step()
    
    def model_zero_grad(self, names=None):
        if names is None:
            names = list(self._optims.keys())
        for name in names:
            self._optims[name].zero_grad()
    
    def model_backward(self, loss, retain_graph=False):
        loss = self.adjust_loss(loss)
        self.detect_anomaly(loss)
        loss.backward(retain_graph=retain_graph)
    
    def model_update(self, names=None):
        if names is None:
            names = list(self._optims.keys())
        for name in names:
            self._optims[name].step()
    
    def model_backward_and_update(self, loss, names=None, retain_graph=False):
        self.model_backward(self.adjust_loss(loss), retain_graph)
        if self.is_update_step:
            self.model_update(names)
            self.update_lr(names)
            self.model_zero_grad(names)
    
    def detect_anomaly(self, loss):
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Loss is infinite or NaN!")
    
    def adjust_loss(self, loss):
        if self.n_gpu > 1:
            loss = loss.mean()
        if self.training_args.gradient_accumulation_steps > 1:
            loss /= self.training_args.gradient_accumulation_steps
        return loss
    
    def joint_loss(self, main_losses, aux_losses=None, lmbda=1.0):
        if not isinstance(main_losses, list):
            main_losses = [main_losses,]
        
        temp_losses = list()
        for l in main_losses:
            if isinstance(l, tuple):
                l, scale = l
            else:
                scale = 1. / len(main_losses)
            l *= scale
            temp_losses.append(l)
        
        loss = sum(temp_losses)
        
        if aux_losses is None:
            aux_losses = []
        if not isinstance(aux_losses, list):
            aux_losses = [aux_losses]
        
        temp_losses = list()
        for l in aux_losses:
            if isinstance(l, tuple):
                l, scale = l
            else:
                scale = 1. / len(aux_losses)
            l *= scale
            temp_losses.append(l)
        
        if temp_losses:
            aux_loss = lmbda * sum(temp_losses)
            loss += aux_loss
        
        return loss
    
    def clip_grad_norm_(self, params):
        nn.utils.clip_grad_norm_(params, self.training_args.max_grad_norm)
    
    def set_common_train_attr(self):
        self.global_step = 0
        self.num_batches_seen = 0
        self.best_metric = None
        num_train_epochs = int(self.training_args.num_train_epochs)
        self.train_iterator = trange(0, num_train_epochs, desc="Epoch")
        self.stats = {
            "batch_train_loss": list(),
            "epoch": list(),
            "step_eval_log": list()
        }
    
    def load_train_data(self, *args, **kwargs):
        assert hasattr(self, "data"), "Can't find data source to load from"
        train_data = self.data.get_train_data(*args, **kwargs)
        assert len(train_data) == 3, "Expected a tuple of 3"
        self.train_data = train_data[0]
        self.train_dataloader_length = train_data[1]
        self.num_train_examples = train_data[2]
    
    def load_eval_data(self, train_datasets=list()):
        assert hasattr(self, "data"), "Can't find data source to load from"
        # Additional dev, test, zero-shot datasets go here and will not be used for training
        # NOTE ``self.validation_datasets`` is the collection used for tracking best model
        eval_datasets = self.exp_args.eval_datasets
        if eval_datasets is None:
            eval_datasets = set()
        else:
            eval_datasets = set(eval_datasets)
        eval_datasets.update(train_datasets)
        eval_datasets = sorted(list(eval_datasets))
        
        # Following two arguments only matter in multi-datasets setting
        if hasattr(self.exp_args, "use_all_shared_clf"):
            all_shared = self.exp_args.use_all_shared_clf
        else:
            all_shared = False
        if hasattr(self.exp_args, "use_private_clf"):
            private = self.exp_args.use_private_clf
        else:
            private = False
        
        # Both are mappings of dataset unique name to NERDataset and the id2label
        self.eval_data, self.test_data = self.data.get_eval_and_test_dataloaders(
            eval_datasets,
            do_eval=self.training_args.do_eval,
            do_predict=self.training_args.do_predict,
            all_shared=all_shared,
            private=private
        )
    
    @abstractmethod
    def load_data(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def batch_to_device(self, *args, **kwargs):
        pass
    
    def before_epoch(self):
        pass
    
    @abstractmethod
    def run_epoch(self, *args, **kwargs):
        pass
    
    def after_epoch(self):
        model = self._models["model"]
        if self.model_args.evaluate_during_training:
            results = self.run_eval(
                model, 
                loss_only=False, 
                save_predictions=False, 
                verbose=True,
                eval_datasets=self.validation_datasets
            )
            results_list = list()
            for name in results.keys():
                results_list.append(results[name])
            self.stats["epoch"].append(results_list)
            self.check_and_save_best_model(model, results)
    
    @abstractmethod
    def before_train(self, *args, **kwargs):
        pass
    
    def train(self):
        self.set_common_train_attr()
        self.before_train_log(self.total_train_batch_size, self.t_total)
        for name, model in self._models.items():
            self._models[name] = self.model_to_gpu(model)
        
        for self.epoch in self.train_iterator:
            self.epoch_iterator = trange(0, self.train_dataloader_length, desc="Iteration")
            self.num_batches = len(self.epoch_iterator)
            
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
            
            if self.is_max_step:
                self.train_iterator.close()
                break
        
        self.dump_stats(self.stats)
        
        return self.global_step, self.stats
    
    def register_model_validation_datasets(self, unique_names):
        if isinstance(unique_names, str):
            unique_names = [unique_names,]
        self.validation_datasets = unique_names
    
    def before_train_log(self, total_train_batch_size, t_total=-1):
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_train_examples)
        logger.info("  Num Epochs = %d", self.training_args.num_train_epochs)
        logger.info(
            "  Instantaneous batch size per device = %d",
            self.training_args.per_device_train_batch_size
        )
        logger.info(
            "  Total train batch size (w. parallel & accumulation) = %d",
            total_train_batch_size
        )
        logger.info(
            "  Gradient Accumulation steps = %d",
            self.training_args.gradient_accumulation_steps
        )
        logger.info("  Total optimization steps = %d", t_total)
    
    def do_logging_step(self, model, logs, evaluate=False):
        if self.training_args.logging_steps > 0 and \
           self.global_step % self.training_args.logging_steps == 0 and \
           self.global_step > 1:
            self.log(logs)
            if self.model_args.evaluate_during_training and evaluate:
                results = self.run_eval(model, eval_datasets=self.validation_datasets)
                self.stats["step_eval_log"].append(results)
                self.check_and_save_best_model(model, results)
    
    def do_save_step(self, model):
        if self.training_args.save_steps > 0 and \
           self.global_step % self.training_args.save_steps == 0 and \
           self.global_step > 1:
            # Save model checkpoint
            output_dir = os.path.join(
                self.training_args.output_dir,
                "checkpoint-{}".format(self.global_step)
            )
            self.save_model(model, output_dir)
    
    def log(self, logs):
        logger.info("***** Training log *****")
        for key in sorted(logs.keys()):
            logger.info("  %s = %s", key, str(logs[key]))
    
    def dump_stats(self, stats):
        stats_json_file = os.path.join(self.training_args.output_dir, "stats.json")
        with open(stats_json_file, "w") as wf:
            json.dump(stats, wf, indent=2)
    
    def check_and_save_best_model(self, model, results):
        eval_metric = list()
        for unique_name in results:
            eval_metric.append(results[unique_name][self.model_args.valid_metric])
        eval_metric = sum(eval_metric) / len(eval_metric)
        
        if self.best_metric is None:
            self.best_metric = eval_metric
        
        # Save best checkpoint
        if self.model_args.valid_metric == "loss":
            flag = eval_metric <= self.best_metric
        else:
            flag = eval_metric >= self.best_metric

        if flag:
            best_output_dir = os.path.join(self.training_args.output_dir, "best")
            self.save_model(model, best_output_dir)
            for unique_name in results:
                self.evaluator.save_results(
                    results[unique_name], best_output_dir, unique_name, "dev"
                )
            self.best_metric = eval_metric
    
    def run_eval(self, model, **kwargs):
        output_dir = kwargs.get("output_dir", self.training_args.output_dir)
        eval_datasets = kwargs.get("eval_datasets", None)
        if eval_datasets is not None:
            eval_data = {
                unique_name: data 
                for unique_name, data in self.eval_data.items() 
                if unique_name in eval_datasets
            }
            for unique_name in eval_datasets:
                shared_unique_name = unique_name + ".shared"
                if shared_unique_name in self.eval_data:
                    eval_data[shared_unique_name] = self.eval_data[shared_unique_name]
        else:
            eval_data = self.eval_data
        results = self.evaluator.run_eval_on_collection(
            model=model,
            collection=eval_data,
            output_dir=output_dir,
            split="dev",
            input_dir=self.exp_args.preprocessed_data_dir,
            loss_only=kwargs.pop("loss_only", False),
            save_predictions=kwargs.pop("save_predictions", False)
        )
        return results
    
    def run_predict(self, model, **kwargs):
        output_dir = kwargs.get("output_dir", self.training_args.output_dir)
        results = self.evaluator.run_eval_on_collection(
            model=model,
            collection=self.test_data,
            output_dir=output_dir,
            split="test",
            input_dir=self.exp_args.preprocessed_data_dir,
            loss_only=kwargs.pop("loss_only", False),
            predict_only=kwargs.pop("predict_only", False),
            save_predictions=kwargs.pop("save_predictions", True)
        )
        return results
    
    def eval_and_test_checkpoints(self):
        # Initialization for evaluation
        if self.model_args.init_checkpoint:
            checkpoint = self.model_args.init_checkpoint
        else:
            checkpoint = self.training_args.output_dir
        
        best_path = os.path.join(self.training_args.output_dir, "best")
        checkpoints = [checkpoint]
        
        # Check for best checkpoint and add it as well
        if os.path.exists(best_path) and best_path != checkpoint:
            checkpoints.append(best_path)
        
        return checkpoints
    
    def do_eval(self, model_class):
        if not self.training_args.do_eval:
            return
        
        checkpoints = self.eval_and_test_checkpoints()
        for ckpt in checkpoints:
            logger.info("Evaluate the following checkpoint: %s", ckpt)
            model = self.load_or_create_model(model_class, init_checkpoint=ckpt)
            model = self.model_to_gpu(model)
            self.run_eval(model, output_dir=ckpt)
            time.sleep(3) # give user time to read the screen
    
    def do_predict(self, model_class):
        if not self.training_args.do_predict:
            return
        
        checkpoints = self.eval_and_test_checkpoints()
        for ckpt in checkpoints:
            logger.info("Predicting with the following checkpoint: %s", ckpt)
            model = self.load_or_create_model(model_class, init_checkpoint=ckpt)
            model = self.model_to_gpu(model)
            self.run_predict(model, output_dir=ckpt, loss_only=False, save_predictions=True)
            time.sleep(3) # give user time to read the screen
    
    def do_train(self, model_class):
        if not self.training_args.do_train:
            return
        
        model = self.load_or_create_model(
            model_class=model_class,
            model_name_or_path=self.model_args.model_name_or_path,
            init_checkpoint=self.model_args.init_checkpoint,
            **self.config_kwargs
        )
        
        # Add extra network layers between encoder and classifier(s)
        if hasattr(self, "xnet_class"):
            if hasattr(self, "xnet_kwargs"):
                xnet_kwargs = self.xnet_kwargs
            else:
                xnet_kwargs = dict()
            xnet = self.xnet_class(model.config, **xnet_kwargs)
            model.add_xnet(xnet)
        
        self.model = model
        self.before_train()
        
        logger.info("Training/evaluation parameters %s", self.training_args)
        global_step, stats = self.train()
        tr_loss = sum(stats["batch_train_loss"]) / global_step
        logger.info(
            "\n\nTraining completed. Do not forget to share your model on "
            "huggingface.co/models =)\n\n"
        )
        logger.info(" global_step = %s, tr_loss = %s", global_step, tr_loss)
        time.sleep(3) # give user time to read the screen
        
        self.save_model(
            model=self.model,
            output_dir=self.training_args.output_dir, 
            training_args=self.training_args,
            model_args=self.model_args,
            exp_args=self.exp_args
        )
        # If training was started with initial checkpoint, we should
        # clear it so that evaluation (if any) can happen on now trained
        if self.model_args.init_checkpoint:
            self.model_args.init_checkpoint = None
    
    def run(self, model_class):
        self.do_train(model_class)
        self.do_eval(model_class)
        self.do_predict(model_class)
