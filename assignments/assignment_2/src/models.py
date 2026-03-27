import torch
from enum import Enum


class Activation(Enum):
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"


_ACTIVATION_MODULES = {
    Activation.RELU: torch.nn.ReLU,
    Activation.TANH: torch.nn.Tanh,
    Activation.SIGMOID: torch.nn.Sigmoid,
}


class MLP(torch.nn.Module):
    def __init__(
            self, 
            input_size: int, 
            hidden_sizes: list[int], 
            output_size: int, 
            activation: Activation):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(torch.nn.Linear(prev_size, hidden_size))
            layers.append(_ACTIVATION_MODULES[activation]())
            prev_size = hidden_size
        layers.append(torch.nn.Linear(prev_size, output_size))
        self.network = torch.nn.Sequential(*layers)

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


# Utility functions

def save_model(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)

def load_model(model: torch.nn.Module, path: str):
    model.load_state_dict(torch.load(path))