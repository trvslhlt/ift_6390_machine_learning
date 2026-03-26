import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

URL = "https://raw.githubusercontent.com/ddidacus/refgen/main/data/chedl_thermo_properties.csv"
SEED = 42


def smiles_to_features(smiles: str, char_to_idx: dict) -> np.ndarray:
    counts = np.zeros(len(char_to_idx), dtype=np.float32)
    for c in smiles:
        if c in char_to_idx:
            counts[char_to_idx[c]] += 1
    length = np.array([len(smiles)], dtype=np.float32)
    return np.concatenate([counts, length])


def get_smiles_dataloaders(
        batch_size: int = 64,
        test_proportion: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    F_LABEL = 'Tc'

    df_raw = pd.read_csv(URL)
    df = df_raw[df_raw[F_LABEL].notna() & df_raw["SMILES"].notna()].copy()
    df = df[["SMILES", F_LABEL]].reset_index(drop=True)
    df.columns = ["smiles", "target"]

    train_df, val_df = train_test_split(df, test_size=test_proportion, random_state=SEED)

    vocab = sorted(set("".join(df["smiles"])))
    char_to_idx = {c: i for i, c in enumerate(vocab)}

    X_train = np.stack([smiles_to_features(s, char_to_idx) for s in train_df["smiles"]])
    X_val = np.stack([smiles_to_features(s, char_to_idx) for s in val_df["smiles"]])
    y_train = train_df["target"].values.astype(np.float32).reshape(-1, 1)
    y_val = val_df["target"].values.astype(np.float32).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
        batch_size=batch_size)

    return train_loader, val_loader
