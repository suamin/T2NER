# -*- coding: utf-8 -*-

from dataclasses import dataclass, field

from ..base import ArgumentsBase
from ..train import BaseTrainer
from ..data import MultiData


class MultiTaskNERTrainer(BaseTrainer):
    
    def __init__(self, training_args, model_args, exp_args):
        super().__init__(training_args, model_args, exp_args)
        self.load_data()
    
    def load_data(self):
        train_metadata_list = [
            self.parse_input_to_dataset_metadata(item) 
            for item in self.exp_args.train_datasets
        ]
        train_datasets = [metadata["name"] for metadata in train_metadata_list]
        
        if self.exp_args.valid_datasets:
            valid_datasets = [
                self.parse_input_to_dataset_metadata(item)["name"]
                for item in self.exp_args.valid_datasets 
                if item in train_datasets # make sure valid split is in train set
            ]
        else:
            valid_datasets = train_datasets
        
        data_kwargs = self.data_kwargs
        data_dir = data_kwargs.pop("preprocessed_data_dir")
        
        self.data = MultiData(data_dir, train_metadata_list, **data_kwargs)
        self.data.train_metadata = dict(
            max_examples=self.exp_args.max_num_train_examples,
            private=self.exp_args.use_private_clf
        )
        self.data.register_training_datasets_with_labels(train_datasets)
        self.register_model_validation_datasets(valid_datasets)
        self.load_eval_data(train_datasets=train_datasets)
        
        # Config kwargs that will be hooked to model config file
        self.config_kwargs.update(dict(
            multidata_type=self.data.type,
            heads_info=self.data.heads_info,
            lang2id=self.data.lang2id,
            domain2id=self.data.domain2id,
            langs=self.data.langs,
            domains=self.data.domains,
            private_clf=self.exp_args.use_private_clf,
            shared_clf=self.exp_args.use_shared_clf,
            all_shared=self.exp_args.use_all_shared_clf,
            num_shared_labels=self.data.total_shared_labels,
            shared_label2id=self.data.shared_label2id,
            shared_id2label=self.data.shared_id2label,
            ignore_metadata=self.exp_args.ignore_metadata,
            add_lang_clf=self.exp_args.add_lang_clf,
            add_domain_clf=self.exp_args.add_domain_clf,
            add_type_clf=self.exp_args.add_type_clf,
            add_all_outside_clf=self.exp_args.add_all_outside_clf,
            add_lm=self.exp_args.add_lm,
            pooling=self.exp_args.pooling
        ))
    
    def batch_to_device(self, inputs):
        for key in inputs:
            for tensor_name, tensor in inputs[key].items():
                inputs[key][tensor_name] = tensor.to(self.device)
        return inputs
    
    def before_train(self):
        self.load_train_data()
        self.model_optim_config = dict(
            model=self.model, 
            optim="adamw", 
            lr_scheduler=self.exp_args.lr_scheduler
        )
        self.configure_optimization()
    
    def run_epoch(self):
        train_dataloader = self.train_data["ner"]
        train_iter = self.data.to_iter(train_dataloader)
        
        if "lm" in self.train_data:
            train_lm_dataloader = self.train_data["lm"]
            train_lm_iter = self.data.to_iter(train_lm_dataloader)
            has_lm = True
        else:
            has_lm = False
        
        model = self.model
        
        for self.step in self.epoch_iterator:
            inputs = next(train_iter)
            inputs = self.batch_to_device(inputs)
            model.train()
            
            outputs, aux_outputs = model(inputs)
            main_losses = [outputs[pred_layer]["ner"]["loss"] for pred_layer in outputs]
            
            if aux_outputs:
                aux_losses = [item["loss"] for item in aux_outputs.values()]
            else:
                aux_losses = []
            
            if has_lm:
                lm_inputs = next(train_lm_iter)
                lm_inputs = self.batch_to_device(lm_inputs)
                lm_outputs = model(lm_inputs, is_lm=True)
                lm_loss = sum([outputs[pred_layer]["loss"] for pred_layer in lm_outputs])/len(lm_outputs)
                aux_losses.append(lm_loss)
            
            if not aux_losses:
                aux_losses = None
            
            loss = self.joint_loss(main_losses, aux_losses, lmbda=self.exp_args.aux_lmbda)
            
            self.model_backward(loss)
            self.stats["batch_train_loss"].append(self.adjust_loss(loss).item())
            self.num_batches_seen += 1
            
            if self.is_update_step:
                self.clip_grad_norm_(model.parameters())
                self.model_update()
                self.global_step += 1
                self.update_lr()
                self.model_zero_grad()
                
                logs = {}
                logs["loss"] = sum(self.stats["batch_train_loss"]) / self.num_batches_seen
                logs["epoch"] = self.progress
                logs["step"] = self.global_step
                
                self.do_logging_step(model, logs, evaluate=True)
                self.do_save_step(model)
            
            if self.is_max_step:
                self.epoch_iterator.close()
                break


@dataclass
class MultiTaskNERArguments(ArgumentsBase):
    """
    Experiment specific arguments.
    """
    preprocessed_data_dir: str = field(
        metadata={"help": "The data directory created by preprocessing script."}
    )
    train_datasets: list = field(
        metadata={"help": "NER training dataset."}
    )
    valid_datasets: list = field(
        default=None,
        metadata={"help": "Validation datasets used for tracking performance (default = all train set)."}
    )
    eval_datasets: list = field(
        default=None,
        metadata={"help": "Additional datasets to evaluate the trained model in zero-shot setting."}
    )
    use_private_clf: bool = field(
        default=False,
        metadata={"help": "Use different classification layers for same label set datasets."}
    )
    use_shared_clf: bool = field(
        default=True,
        metadata={"help": "Use same classification layers for same label set datasets."}
    )
    use_all_shared_clf: bool = field(
        default=False,
        metadata={"help": "Add shared prediction layer for all the datasets in training."}
    )
    ignore_metadata: bool = field(
        default=True,
        metadata={"help": "Whether to ignore language and domain information."}
    )
    add_lang_clf: bool = field(
        default=False,
        metadata={"help": "Whether to add language identification task."}
    )
    add_domain_clf: bool = field(
        default=False,
        metadata={"help": "Whether to add domain classification task."}
    )
    add_type_clf: bool = field(
        default=False,
        metadata={"help": "Whether to add entity type classifier."}
    )
    add_all_outside_clf: bool = field(
        default=False,
        metadata={"help": "Whether to add classifier predicting if sentence has all O tags or not."}
    )
    add_lm: bool = field(
        default=False,
        metadata={"help": "Add joint language modeling head during training."}
    )
    pooling: str = field(
        default="mean",
        metadata={"help": "Add a pooling strategy (only effective when pooled output is used)."}
    )
    aux_lmbda: float = field(
        default=1.0,
        metadata={"help": "Lambda parameter to control effect of auxiliary losses."}
    )
    max_num_train_examples: int = field(
        default=-1,
        metadata={"help": "Maximum no. of training examples to consider (-1 = consider all)."}
    )
    lr_scheduler: str = field(
        default="linear",
        metadata={"help": "Which learning rate scheduler to use."}
    )
