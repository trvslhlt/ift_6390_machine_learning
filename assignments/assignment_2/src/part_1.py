from data_prep import get_smiles_dataloaders
from models import MLP, Activation
from training import train
import utils

import torch
from torch import optim
from dataclasses import dataclass


ARTIFACT_PREFIX = 'exp_1'


@dataclass
class Output:
    logs_dir: str
    models_dir: str


@dataclass
class Hyperparams:
    epochs: int
    learning_rate: float
    momentum: float

    hidden_sizes: list[int]
    activation_fn: Activation
    dropout: float | None
    batch_norm: bool


@dataclass
class RunConfig:
    device: str
    output: Output
    hyperparams: Hyperparams


def run_1(config: RunConfig):
    hp = config.hyperparams
    
    training_dataloader, validation_dataloader, target_stats = get_smiles_dataloaders()
    feature_count = training_dataloader.dataset.tensors[0].shape[1]
    
    logs = {
        'experiment': {
            'model': {},
            'hyperparams': {
                'epochs': hp.epochs,
                'learning_rate': hp.learning_rate,
                'momentum': hp.momentum,
            },
            'target_stats': target_stats,
        }, 'epoch': [], 'batch': []}


    model = MLP(
        input_size=feature_count,
        hidden_sizes=hp.hidden_sizes,
        output_size=1, # regression, single value prediction
        activation=Activation.RELU)

    def on_batch_end(model, **kwargs):
        logs['batch'].append(kwargs)
        print(f'epoch: {kwargs["epoch"]}, batch_idx: {kwargs["batch"]}')

    def on_epoch_end(model, **kwargs):
        logs['experiment']['model'] = model.describe()
        logs['epoch'].append(kwargs)

    train(
        model=model,
        epochs=hp.epochs,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optim.SGD(model.parameters(), lr=hp.learning_rate, momentum=hp.momentum),
        loss_fn=torch.nn.MSELoss(),
        device=config.device,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
    )

    artifact_id = f'{ARTIFACT_PREFIX}_1'
    utils.save_logs(logs, config.output.logs_dir, f'{artifact_id}__{utils.file_timestamp()}')
    utils.save_model(model, config.output.models_dir, f'{artifact_id}__{utils.file_timestamp()}')
