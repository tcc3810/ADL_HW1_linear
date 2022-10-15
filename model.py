from typing import Dict

import torch
import torch.nn as nn
from torch.nn import Embedding

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,   # word embeddings
        hidden_size: int,           # 每層有幾個節點
        num_layers: int,            # 有幾層
        dropout: float,             # 改變overfitting (softmax). 訓練才會用到, 測試不用
        bidirectional: bool,        # 是否為雙向
        num_class: int,             # 總共有幾類
    ) -> None:
        
        super(SeqClassifier, self).__init__()
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        # TODO: model architecture
        self.lstm = torch.nn.LSTM(
            input_size = embeddings.size(dim = 1),
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
        )
        if bidirectional:
            self.linear = nn.Linear(hidden_size * 4 , num_class)
        else:
            self.linear = nn.Linear(hidden_size * 2 , num_class)
        
    
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:    
        # batch:{ "text" : [Tensor_A, B, C, ...] , "intent" :[Tensor_a, b, c, ...] } return: {"text" : Tensor , "intent" : Tensor , ...}
        # TODO: implement model forward
        
        # batch = [256 , 128]
        # embedding = [256, 128, 300]
        embeddings = self.embed(batch)

        # output = [128, 256, 1024]
        output , _ = self.lstm(embeddings.permute([1 , 0 , 2]))

        # out = [256, 2048]
        out = torch.cat([output[0] , output[-1]] , dim=1)
        
        # out = [256 , 150]
        out = self.linear(out)
        
        return out

class SeqTagger(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,   
        hidden_size: int,           
        num_layers: int,            
        dropout: float,            
        bidirectional: bool,      
        num_class: int,
    ) -> None:
        
        super(SeqTagger, self).__init__()
        
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        
        # TODO: model architecture
        self.lstm = torch.nn.LSTM(
            input_size = embeddings.size(dim = 1),
            hidden_size = hidden_size,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional,
            batch_first = True
        )
        
        self.activate = nn.ReLU()
        self.layernorm = nn.LayerNorm(embeddings.shape[1])
        
        if bidirectional:
            self.linear1 = nn.Linear(hidden_size * 2 , hidden_size)
            self.linear = nn.Linear(hidden_size * 2 , num_class)
            self.batchnorm1 = nn.BatchNorm1d(num_features = hidden_size * 2)
        else:
            self.linear1 = nn.Linear(hidden_size * 1 , hidden_size)
            self.linear = nn.Linear(hidden_size * 1 , num_class)
            self.batchnorm1 = nn.BatchNorm1d(num_features = hidden_size * 1)
        
        self.linear2 = nn.Linear(hidden_size , num_class)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size)

    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError            
    
    def forward(self, batch) -> Dict[str, torch.Tensor]:    
        # TODO: implement model forward
        
        '''
        # method 1:
        embeddings = self.embed(batch)
        embeddings = self.layernorm(embeddings)
        outputs , _ = self.lstm(embeddings)

        out = self.batchnorm1(outputs.permute([0 , 2 , 1]))
        out = self.linear1(out.permute([0 , 2 , 1]))
        out = self.activate(out)

        out = self.batchnorm2(out.permute([0 , 2 , 1]))
        out = self.linear2(out.permute([0 , 2 , 1]))
        out = self.activate(out)

        out = out.permute([0 , 2 , 1])
        return out
        '''
        
        # method 2 :
        embeddings = self.embed(batch)
        outputs , _ = self.lstm(embeddings)
        out = self.linear(outputs)

        out = out.permute([0 , 2 , 1])
        return out

