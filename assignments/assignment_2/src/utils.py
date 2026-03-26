from datetime import datetime
import json
import os

import torch


def file_timestamp():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def save_logs(logs: dict, logs_dir: str, filename: str):
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, f'{filename}.json'), 'w') as f:
       json.dump(logs, f, indent=4)


def save_model(model: torch.nn.Module, models_dir: str, filename: str):
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f'{filename}.pt')
    torch.save(model.state_dict(), filepath)