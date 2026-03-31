from data_prep import get_smiles_dataloaders
from models import MLP, Activation, Initialization
from training import train
import utils
from utils import Output

from dataclasses import dataclass, asdict

import torch


@dataclass
class Hyperparams:
    epochs: int
    hidden_sizes: list[int]
    activation: str
    optimizer: str
    lr: float
    momentum: float | None
    dropout: float | None = None
    batch_norm: bool = False
    initialization: str | None = None


@dataclass
class RunConfig:
    device: str
    output: Output
    hyperparams: Hyperparams
    exp_id: str


def run_1(config: RunConfig):
    hp = config.hyperparams
    
    training_dataloader, validation_dataloader, target_stats = get_smiles_dataloaders()
    feature_count = training_dataloader.dataset.tensors[0].shape[1]
    
    start_time = utils.unix_timestamp()
    logs = {
        'metadata': {
            'start_time': utils.iso_timestamp(),
            'device': str(config.device),
            'exp_id': config.exp_id,
        },
        'experiment': {
            'model': {},
            'hyperparams': asdict(hp),
            'target_stats': target_stats,
            'dataset': {
                'train_size': len(training_dataloader.dataset),
                'val_size': len(validation_dataloader.dataset),
                'feature_count': feature_count,
                'batch_size': training_dataloader.batch_size,
            },
        }, 'epoch': [], 'batch': []}

    initialization = None if hp.initialization is None else Initialization(hp.initialization)
    dropout = 0 if hp.dropout is None else hp.dropout
    model = MLP(
        input_size=feature_count,
        hidden_sizes=hp.hidden_sizes,
        output_size=1, # regression, single value prediction
        activation=Activation(hp.activation),
        initialization=initialization,
        dropout=dropout,
        batch_norm=hp.batch_norm,
    )
    optimizer = utils.get_optimizer(hp.optimizer, model.parameters(), hp.lr, hp.momentum)
    logs['experiment']['model'] = model.describe()

    def on_batch_end(**kwargs):
        logs['batch'].append(kwargs)
        print(f'epoch: {kwargs["epoch"]}, batch_idx: {kwargs["batch"]}')

    def on_epoch_end(**kwargs):
        logs['epoch'].append(kwargs)

    train(
        model=model,
        epochs=hp.epochs,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        device=config.device,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
    )

    logs['metadata']['end_time'] = utils.iso_timestamp()
    logs['metadata']['elapsed_time'] = utils.unix_timestamp() - start_time

    utils.save_logs(logs, config.output.logs_dir, f'{config.exp_id}__{utils.file_timestamp()}')
    utils.save_model(model, config.output.models_dir, f'{config.exp_id}__{utils.file_timestamp()}')
