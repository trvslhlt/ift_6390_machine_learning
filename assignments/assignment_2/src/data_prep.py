import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, TensorDataset


URL = 'https://raw.githubusercontent.com/ddidacus/refgen/main/data/chedl_thermo_properties.csv'
SEED = 42


class SmilesIndicesDataset(Dataset):
    '''custom dataset providing indexed smiles character sequences'''
    def __init__(self, smiles_list: list[str], targets: np.ndarray, char_to_idx: dict):
        super().__init__()
        self.sequences = [torch.tensor([char_to_idx[c] for c in s], dtype=torch.long) for s in smiles_list]
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''given a batch of (sequence, target), pads the sequences and notes their original lengths'''
    sequences, targets = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded_sequences, lengths, targets # keep convention of (*inputs, targets) for training loop


def get_smiles_embed_seq_dataloaders(
        batch_size: int = 64,
        test_proportion: float = 0.2,
        standardize_targets: bool = True,
) -> tuple[DataLoader, DataLoader, dict, int, int]:
    '''creates dataloaders for indexed sequences, returns them along with dataset metrics'''
    df = get_smiles_df()

    train_df, val_df = train_test_split(df, test_size=test_proportion, random_state=SEED)

    vocab = sorted(set("".join(df["smiles"])))
    char_to_idx = {c: i + 1 for i, c in enumerate(vocab)} # index from 1 to reserve 0 for padding idx

    y_train = train_df["target"].values.astype(np.float32).reshape(-1, 1)
    y_val = val_df["target"].values.astype(np.float32).reshape(-1, 1)

    y_mean = y_train.mean()
    y_std = y_train.std()
    if standardize_targets:
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
    target_stats = {'mean': float(y_mean), 'std': float(y_std)}

    train_loader = DataLoader(
        SmilesIndicesDataset(train_df['smiles'].tolist(), y_train, char_to_idx),
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn)
    val_loader = DataLoader(
        SmilesIndicesDataset(val_df['smiles'].tolist(), y_val, char_to_idx),
        batch_size=batch_size, 
        collate_fn=collate_fn)

    num_embeddings = len(vocab) + 1 # +1 for padding idx 0
    max_seq_len = max(len(s) for s in df["smiles"])

    return train_loader, val_loader, target_stats, num_embeddings, max_seq_len


def get_smiles_dataloaders(
        batch_size: int = 64,
        test_proportion: float = 0.2,
        standardize_targets: bool = True,
) -> tuple[DataLoader, DataLoader, dict]:
    '''creates dataloaders for count-based features, returns them along with dataset metrics'''
    df = get_smiles_df()

    train_df, val_df = train_test_split(df, test_size=test_proportion, random_state=SEED)

    vocab = sorted(set("".join(df["smiles"])))
    char_to_idx = {c: i for i, c in enumerate(vocab)}

    X_train = np.stack([_smiles_to_features(s, char_to_idx) for s in train_df["smiles"]])
    X_val = np.stack([_smiles_to_features(s, char_to_idx) for s in val_df["smiles"]])
    y_train = train_df["target"].values.astype(np.float32).reshape(-1, 1)
    y_val = val_df["target"].values.astype(np.float32).reshape(-1, 1)

    y_mean = y_train.mean()
    y_std = y_train.std()
    if standardize_targets:
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
    target_stats = {'mean': float(y_mean), 'std': float(y_std)}

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size)

    return train_loader, val_loader, target_stats


def _smiles_to_features(smiles: str, char_to_idx: dict) -> np.ndarray:
    '''Convert a SMILES string to a feature vector by counting the occurrences
    of each character and the length of the string.'''
    counts = np.zeros(len(char_to_idx), dtype=np.float32)
    for c in smiles:
        if c in char_to_idx:
            counts[char_to_idx[c]] += 1
    length = np.array([len(smiles)], dtype=np.float32)
    return np.concatenate([counts, length])


def get_smiles_df():
    '''download smiles data, filter rows and columns, and returns dataframe'''
    F_LABEL = 'Tc'
    F_SMILES = 'SMILES'

    df_raw = pd.read_csv(URL, low_memory=False)
    df = df_raw[df_raw[F_LABEL].notna() & df_raw[F_SMILES].notna()].copy()
    df = df[[F_SMILES, F_LABEL]].reset_index(drop=True)
    df.columns = ['smiles', 'target']

    return df