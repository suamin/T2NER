# -*- coding: utf-8 -*-

import os
import random
import logging
import numpy as np
import torch
import dataclasses
import json

from dataclasses import dataclass
from transformers.file_utils import cached_property


logger = logging.getLogger(__name__)


class TrainInferenceUtils:
    
    @cached_property
    def _setup_devices(self):
        logger.info("PyTorch: setting up devices")
        if hasattr(self, "training_args"):
            no_cuda = self.training_args.no_cuda
        elif hasattr(self, "no_cuda"):
            no_cuda = self.no_cuda
        else:
            no_cuda = False
        if not no_cuda:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            device = torch.device("cpu")
            n_gpu = 0
        return device, n_gpu
    
    @property
    def device(self):
        return self._setup_devices[0]
    
    @property
    def n_gpu(self):
        return self._setup_devices[1]
    
    @property
    def seed(self):
        return self._seed
    
    @seed.setter
    def seed(self, value):
        self.set_seed(value)
        self._seed = value
    
    @property
    def train_batch_size(self):
        if hasattr(self, "training_args"):
            per_device_train_batch_size = self.training_args.per_device_train_batch_size
        else:
            per_device_train_batch_size = self.per_device_train_batch_size
        return per_device_train_batch_size * max(1, self.n_gpu)
    
    @property
    def eval_batch_size(self):
        if hasattr(self, "training_args"):
            per_device_eval_batch_size = self.training_args.per_device_eval_batch_size
        else:
            per_device_eval_batch_size = self.per_device_eval_batch_size
        return per_device_eval_batch_size * max(1, self.n_gpu)
    
    @staticmethod
    def load_or_create_model(
        model_class,
        model_name_or_path=None,
        init_checkpoint=None,
        **config_kwargs
    ):
        if model_name_or_path is None and init_checkpoint is None:
            raise ValueError(
                "For loading or creating a new model model_name_or_path or "
                "init_checkpoint is required. When both are passed, init_checkpoint "
                "takes precedence."
            )
        if init_checkpoint:
            model_name_or_path = init_checkpoint
            logger.info("loading from init_checkpoint = {}".format(model_name_or_path))
        else:
            logger.info("loading from cached model = {}".format(model_name_or_path))
        
        model = model_class.from_pretrained(
            model_name_or_path,
            config_kwargs=config_kwargs
        )
        
        return model
    
    @staticmethod
    def save_model(
        model,
        output_dir,
        training_args=None,
        model_args=None,
        exp_args=None
    ):
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # In all cases (even distributed/parallel), model is always a reference
        # to the model we want to save.
        if hasattr(model, "module"):
            save_model = model.module
        else:
            save_model = model
        save_model.save_pretrained(output_dir)
        
        # Good practice: save your arguments together with the trained model
        for name, args in [
            ("training_args", training_args), 
            ("model_args", model_args),
            ("exp_args", exp_args)
        ]:
            if args is not None:
                args_json = args.to_json_string()
                with open(os.path.join(output_dir, name + ".json"), "w") as wf:
                    wf.write(args_json)
    
    @staticmethod
    def parse_input_to_dataset_metadata(item):
        metadata = dict(
            name=None,
            max_examples=-1,
            drop_last=False,
            forever=False, 
            shuffle=False
        )
        if isinstance(item, str):
            metadata["name"] = item
        elif len(item) > 1:
            metadata["name"] = item[0]
            for i in range(1, len(item)-1):
                if i == 1:
                    metadata["max_examples"] = item[1]
                elif i == 2:
                    metadata["drop_last"] = item[2]
                elif i == 3:
                    metadata["forever"] = item[3]
                elif i == 4:
                    metadata["shuffle"] = item[4]
        else:
            raise ValueError(
                "Unable to parse item `{}` to metadata.".format(item)
            )
        return metadata
    
    @staticmethod
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
    
    @staticmethod
    def freeze(module):
        for params in module.parameters():
            params.requires_grad = False
    
    @staticmethod
    def unfreeze(module):
        for params in module.parameters():
            params.requires_grad = True
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)
    
    @staticmethod
    def detach(tensor):
        return tensor.detach().cpu().numpy()


@dataclass
class ArgumentsBase:
    
    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)
