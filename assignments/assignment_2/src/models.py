import torch
from enum import Enum
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
        super(MLP, self).__init__()

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
            output_size: int
        ):
        super(LSTM, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_size, padding_idx=0)
        self.lstm = torch.nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.linear(h_n.squeeze(0))

    def describe(self) -> dict:
        return {
            'vocab_size': self.embedding.num_embeddings,
            'embedding_size': self.embedding.embedding_dim,
            'hidden_size': self.lstm.hidden_size,
            'output_size': self.linear.out_features,
            'num_params': sum(p.numel() for p in self.parameters()),
        }


def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))