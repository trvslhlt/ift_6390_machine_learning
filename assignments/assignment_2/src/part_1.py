from data_prep import get_smiles_dataloaders
from models import MLP, Activation
from training import train
import utils

import torch
from torch import optim


def run_1_1(device: str, logs_dir: str, models_dir: str):
    epochs = 500
    learning_rate = 1e-3 # 0.001
    momentum = 0.0
    hidden_sizes = [40, 20] # 2 hidden layers
    
    training_dataloader, validation_dataloader, target_stats = get_smiles_dataloaders()
    feature_count = training_dataloader.dataset.tensors[0].shape[1]
    
    logs = {
        'experiment': {
            'model': {},
            'hyperparams': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'momentum': momentum,
            },
            'target_stats': target_stats,
        }, 'epoch': [], 'batch': []}


    model = MLP(
        input_size=feature_count,
        hidden_sizes=hidden_sizes,
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
        epochs=epochs,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum),
        loss_fn=torch.nn.MSELoss(),
        device=device,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
    )

    artifact_id = 'exp_1_1'
    utils.save_logs(logs, logs_dir, f'{artifact_id}__{utils.file_timestamp()}')
    utils.save_model(model, models_dir, f'{artifact_id}__{utils.file_timestamp()}')
