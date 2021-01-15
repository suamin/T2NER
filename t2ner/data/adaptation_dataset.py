# -*- coding: utf-8 -*-

from .base import BaseData


class SimpleAdaptationData(BaseData):
    
    def __init__(self, preprocessed_data_dir, src_metadata, tgt_metadata, **kwargs):
        super().__init__(preprocessed_data_dir, **kwargs)
        self.metadata = self.collect_metadata(src_metadata, tgt_metadata)
    
    def collect_metadata(self, src_metadata, tgt_metadata):
        src_lang, src_domain, src_source = src_metadata["name"].split(".")
        tgt_lang, tgt_domain, tgt_source = tgt_metadata["name"].split(".")
        
        if src_lang == tgt_lang and src_domain != tgt_domain:
            mode = "CD"
        elif src_domain == tgt_domain and src_lang != tgt_domain:
            mode = "CL"
        else:
            raise NotImplementedError
        
        label_list = self.get_label_list(src_metadata["name"])
        type_list = self.get_type_list(src_metadata["name"])
        temp_label_list = self.get_label_list(tgt_metadata["name"])
        
        if label_list != temp_label_list:
            raise NotImplementedError(
                "Currently the src-tgt adaptation is only allowed when label spaces are shared."
            )
        
        metadata = {
            "mode": mode,
            "src": {
                "name": src_metadata["name"].replace(".", "_"),
                "lang": src_lang,
                "domain": src_domain,
                "source": src_source,
                "metadata": src_metadata
            },
            "tgt": {
                "name": tgt_metadata["name"].replace(".", "_"),
                "lang": tgt_lang,
                "domain": tgt_domain,
                "source": tgt_source,
                "metadata": tgt_metadata
            },
            "label_list": label_list,
            "num_labels": len(label_list),
            "type_list": type_list,
            "num_types": len(type_list)
        }
        
        return metadata
    
    @property
    def type(self):
        return self.metadata["mode"]
    
    @property
    def src_lang(self):
        return self.metadata["src"]["lang"]
    
    @property
    def src_domain(self):
        return self.metadata["src"]["domain"]
    
    @property
    def tgt_lang(self):
        return self.metadata["tgt"]["lang"]
    
    @property
    def tgt_domain(self):
        return self.metadata["tgt"]["domain"]
    
    def get_train_data(self, k=1, lm=False, src_lm_dataset=None, tgt_lm_dataset=None):
        train_dataloaders = dict(ner=dict(src=None, tgt=None))
        
        src_metadata = self.metadata["src"]["metadata"]
        tgt_metadata = self.metadata["tgt"]["metadata"]
        
        train_dataloaders["ner"]["src"] = self.get_k_train_dataloaders(src_metadata, k=k)
        train_dataloaders["ner"]["tgt"] = self.get_k_train_dataloaders(tgt_metadata, k=k)
        train_dataloader_len = len(train_dataloaders["ner"]["src"][0])
        num_train_examples = len(train_dataloaders["ner"]["src"][0].dataset)
        
        if lm and src_lm_dataset is not None and tgt_lm_dataset is not None:
            train_dataloaders["lm"] = dict(src=None, tgt=None)
            train_dataloaders["lm"]["src"] = self.get_clm_train_dataloader(src_metadata, src_lm_dataset)
            train_dataloaders["lm"]["tgt"] = self.get_clm_train_dataloader(tgt_metadata, tgt_lm_dataset)
        
        return train_dataloaders, train_dataloader_len, num_train_examples
