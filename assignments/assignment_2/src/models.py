from enum import Enum
import math

import torch
from torch.nn.utils.rnn import pack_padded_sequence


class Activation(Enum):
    RELU = 'relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'

_ACTIVATION_MODULES = {
    Activation.RELU: torch.nn.ReLU,
    Activation.TANH: torch.nn.Tanh,
    Activation.SIGMOID: torch.nn.Sigmoid,
}

class Initialization(Enum):
    HE = 'he'

_INITIALIZATION_MODULES = {
    Initialization.HE: torch.nn.init.kaiming_normal_,
}


class MLP(torch.nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: list[int], 
            output_size: int, 
            activation: Activation,
            initialization: Initialization | None,
            dropout: float,
            batch_norm: bool = False,
        ):
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout)

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(self.dropout)
            layers.append(_ACTIVATION_MODULES[activation]())
            prev_size = hidden_size
        layers.append(torch.nn.Linear(prev_size, output_size))
        self.network = torch.nn.Sequential(*layers)
        self._initialize_params(initialization)


    def describe(self) -> dict:
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'activation': self.activation.value,
            'num_params': sum(p.numel() for p in self.parameters()),
        }

    def forward(self, x):
        return self.network(x)
    
    def _initialize_params(self, initialization: Initialization | None):
        if initialization is None:
            return

        init_fn = _INITIALIZATION_MODULES[initialization]
        for module in self.network.modules():
            if isinstance(module, torch.nn.Linear):
                init_fn(module.weight)
                torch.nn.init.zeros_(module.bias)


class LSTM(torch.nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            hidden_size: int,
            output_size: int,
            dropout_proportion: float,
            is_bidirectional: bool = False,
        ):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=is_bidirectional)
        self.dropout = torch.nn.Dropout(dropout_proportion)
        linear_hidden_size = hidden_size * 2 if is_bidirectional else hidden_size
        self.linear = torch.nn.Linear(linear_hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            h_n = h_n.squeeze(0)
        h_n = self.dropout(h_n)
        return self.linear(h_n)

    def describe(self) -> dict:
        return {
            'num_embeddings': self.embedding.num_embeddings,
            'embedding_size': self.embedding.embedding_dim,
            'hidden_size': self.lstm.hidden_size,
            'output_size': self.linear.out_features,
            'dropout_proportion': self.dropout.p,
            'is_bidirectional': self.lstm.bidirectional,
            'num_params': sum(p.numel() for p in self.parameters()),
        }



class Transformer(torch.nn.Module):

    pos_encoding: torch.Tensor

    def __init__(
            self,
            num_embeddings: int,
            embedding_size: int,
            num_heads: int,
            num_encoder_layers: int,
            output_size: int,
            max_seq_length: int,
            dropout_proportion: float,
        ):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.pos_encoding = self._sinusoidal_encoding(max_seq_length, embedding_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_size, 
            nhead=num_heads, 
            dropout=dropout_proportion)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.linear = torch.nn.Linear(embedding_size, output_size)

    def forward(self, x, _lengths):
        pad_mask = (x == 0)
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        mask = (~pad_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # mean pooling over non-padded tokens
        return self.linear(x)

    def describe(self) -> dict:
        return {
            'num_embeddings': self.embedding.num_embeddings,
            'embedding_size': self.embedding.embedding_dim,
            'num_heads': self.transformer.num_heads,
            'num_encoder_layers': self.transformer.num_encoder_layers,
            'output_size': self.linear.out_features,
            'max_seq_length': self.max_seq_length,
            'dropout_proportion': self.dropout.p,
            'num_params': sum(p.numel() for p in self.parameters()),
        }
    
    def _sinusoidal_encoding(self, max_len, d_model) -> torch.nn.Module:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return torch.nn.Parameter(pe.unsqueeze(0), requires_grad=False)  # (1, max_len, d_model)


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))