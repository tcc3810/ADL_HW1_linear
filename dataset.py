from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab

import numpy as np
import torch

class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance
     
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    
    def collate_fn(self, samples: List[Dict]) -> Dict:  
        # [{'text': A, 'intent': B } , {} , ...]    ->  {'text' : Tensor , 'intent' : Tensor}
        # TODO: implement collate_fn
        text_data = []
        intent_data = []
        idx_data = []
        for sample in samples:
            x = sample['text'].split()
            text_data.append(x)
            try:
                y = self.label2idx(sample['intent'])
                intent_data.append(y)
            except KeyError:
                pass 
            idx_data.append(sample['id'])
        
        text_encode = self.vocab.encode_batch(text_data , self.max_len)
        
        text_tensor = torch.tensor(text_encode , dtype = torch.int)
        intent_tensor = torch.tensor(intent_data , dtype = torch.long) 
        
        return { "text" : text_tensor , "intent" : intent_tensor , "id" : idx_data }
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTaggingClsDataset(Dataset):
    #ignore_idx = -100
    
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance
     
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)
    
    def collate_fn(self, samples):  
        # samples = [{"tokens": [str, ...] , "tags": [str, ...] ,"id" : str} , {} , ...]
        #               ->  {"tokens" : [ [str, str, ... ] , ...] , "tags" : [ [str->int, str->int, ... ] , ...] , "id" : [str, str, ...]}
        # TODO: implement collate_fn
        tokens_data = []
        tags_data = []
        idx_data = []
        len_data = []
        for sample in samples:
            x = sample["tokens"]
            tokens_data.append(x)
            
            try:
                y = []
                for tag in sample["tags"]:
                    y.append(self.label2idx(tag))
                for i in range(self.max_len - len(sample["tags"])):
                    y.append(self.label2idx("O"))
            except KeyError:
                pass
            tags_data.append(y)
            
            idx_data.append(sample["id"])
            len_data.append(len(sample["tokens"]))

        tokens_encode = self.vocab.encode_batch(tokens_data , self.max_len)
        
        tokens_tensor = torch.tensor(tokens_encode , dtype = torch.int)
        tags_tensor = torch.tensor(tags_data , dtype = torch.long) 
        
        return { "tokens" : tokens_tensor , "tags" : tags_tensor , "id" : idx_data , "len" : len_data }
    
    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

