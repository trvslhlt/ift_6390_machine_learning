from attr import dataclass, asdict
import torch

from data_prep import get_smiles_embed_seq_dataloaders
from models import LSTM, Transformer
from training import train
import utils
from utils import Output


@dataclass
class Hyperparams:
    epochs: int
    embedding_size: int
    optimizer: str
    lr: float
    hidden_size: int | None = None
    dropout: float | None = None
    is_bidirectional: bool = False
    num_heads: int | None = None
    num_encoder_layers: int | None = None


@dataclass
class RunConfig:
    device: str
    output: Output
    hyperparams: Hyperparams
    exp_id: str
    model: str


def run_2(config: RunConfig):
    print(f'Running with config: {config}')

    hp = config.hyperparams
    
    (
        training_dataloader, 
        validation_dataloader, 
        target_stats, 
        num_embeddings, 
        max_seq_len
    ) = get_smiles_embed_seq_dataloaders()

    start_time = utils.unix_timestamp()
    logs = {
        'metadata': {
            'start_time': utils.iso_timestamp(),
            'device': str(config.device),
            'exp_id': config.exp_id,
            'model': config.model,
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
    
    model = _get_model(config.model, hp, num_embeddings, max_seq_len)
    optimizer = utils.get_optimizer(hp.optimizer, model.parameters(), hp.lr, 0)
    logs['experiment']['model'] = model.describe() # type: ignore

    def on_batch_end(**kwargs):
        logs['batch'].append(kwargs)

    def on_epoch_end(**kwargs):
        logs['epoch'].append(kwargs)
        print(f'epoch: {kwargs["epoch"]}, train_loss: {kwargs["train_loss"]:.4f}, val_loss: {kwargs["val_loss"]:.4f}')

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


def _get_model(model: str, hp: Hyperparams, num_embeddings: int, max_seq_length: int) -> torch.nn.Module:
    dropout = hp.dropout if hp.dropout is not None else 0.0
    if model == 'lstm':
        return LSTM(
            num_embeddings=num_embeddings,
            embedding_size=hp.embedding_size,
            hidden_size=hp.hidden_size,
            output_size=1,
            dropout_proportion=dropout,
            is_bidirectional=hp.is_bidirectional,
        )
    elif model == 'transformer':
        return Transformer(
            num_embeddings=num_embeddings,
            embedding_size=hp.embedding_size,
            num_heads=hp.num_heads if hp.num_heads is not None else 4,
            num_encoder_layers=hp.num_encoder_layers if hp.num_encoder_layers is not None else 2,
            output_size=1,
            max_seq_length=max_seq_length,
            dropout_proportion=dropout,
        )
    else:
        raise ValueError(f'unknown model type: "{model}"')