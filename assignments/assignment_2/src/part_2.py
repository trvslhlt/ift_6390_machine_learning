from attr import dataclass, asdict
import torch

from data_prep import get_smiles_embed_seq_dataloaders
from models import LSTM
from training import train
import utils
from utils import Output



@dataclass
class Hyperparams:
    epochs: int
    embedding_size: int
    hidden_size: int
    optimizer: str
    lr: float
    momentum: float | None = None


@dataclass
class RunConfig:
    device: str
    output: Output
    hyperparams: Hyperparams
    exp_id: str


def run_2(config: RunConfig):
    print(f'Running with config: {config}')
    # SMILES → character embedding → LSTM → final state → linear layer → prediction

    hp = config.hyperparams
    
    training_dataloader, validation_dataloader, target_stats, num_embeddings = get_smiles_embed_seq_dataloaders()
    
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
                'feature_count': num_embeddings,
                'batch_size': training_dataloader.batch_size,
            },
        }, 'epoch': [], 'batch': []}
    
    model = LSTM(
        num_embeddings=num_embeddings,
        embedding_size=hp.embedding_size,
        hidden_size=hp.hidden_size,
        output_size=1 # regression, single value prediction
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
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_fn=torch.nn.MSELoss(),
        device=config.device,
        epochs=hp.epochs,
        on_batch_end=on_batch_end,
        on_epoch_end=on_epoch_end,
    )

    logs['metadata']['end_time'] = utils.iso_timestamp()
    logs['metadata']['elapsed_time'] = utils.unix_timestamp() - start_time

    utils.save_logs(logs, config.output.logs_dir, f'{config.exp_id}__{utils.file_timestamp()}')
    utils.save_model(model, config.output.models_dir, f'{config.exp_id}__{utils.file_timestamp()}')