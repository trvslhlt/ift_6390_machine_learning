from dataclasses import dataclass
from datetime import datetime
import json
import os
import time

import torch
from torch import optim


@dataclass
class Output:
    logs_dir: str
    models_dir: str


def file_timestamp() -> str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def iso_timestamp() -> str:
    return datetime.now().isoformat()


def unix_timestamp() -> float:
    return time.time()


def save_logs(logs: dict, logs_dir: str, filename: str):
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, f'{filename}.json'), 'w') as f:
       json.dump(logs, f, indent=4)


def save_model(model: torch.nn.Module, models_dir: str, filename: str):
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f'{filename}.pt')
    torch.save(model.state_dict(), filepath)

def get_optimizer(name: str, params, lr: float | None, momentum: float | None) -> optim.Optimizer:
    lr = 0 if lr is None else lr
    momentum = 0 if momentum is None else momentum

    if name == 'sgd':
        return optim.SGD(params, lr, momentum)
    elif name == 'adam':
        return optim.Adam(params, lr)
    else:
        raise Exception(f'optimizer not supported: "{name}"')
