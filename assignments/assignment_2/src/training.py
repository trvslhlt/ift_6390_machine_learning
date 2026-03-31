import time
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader


def train(
        model: torch.nn.Module,
        epochs: int,
        training_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: torch.device,
        on_batch_end: Optional[Callable] = None,
        on_epoch_end: Optional[Callable] = None):
    model.to(device)
    train_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch_idx, (X, y) in enumerate(training_dataloader):
            batch_start = time.perf_counter()
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            grad_norms = _layer_grad_norms(model)
            optimizer.step()
            train_loss_sum += loss.item() * X.size(0)
            train_count += X.size(0)
            if on_batch_end:
                on_batch_end(
                    epoch=epoch, 
                    batch=batch_idx, 
                    loss=loss.item(),
                    grad_norms=grad_norms,
                    batch_time=time.perf_counter() - batch_start,
                    elapsed=time.perf_counter() - train_start)

        model.eval()
        train_preds = []
        train_targets = []
        val_loss_sum = 0.0
        val_count = 0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X, y in training_dataloader:
                X, y = X.to(device), y.to(device)
                train_preds.append(model(X))
                train_targets.append(y)
            for X, y in validation_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss_sum += loss.item() * X.size(0)
                val_count += X.size(0)
                val_preds.append(pred)
                val_targets.append(y)

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        train_r2 = _r2_score(torch.cat(train_preds), torch.cat(train_targets))
        val_r2 = _r2_score(torch.cat(val_preds), torch.cat(val_targets))
        if on_epoch_end:
            on_epoch_end(
                epoch=epoch,
                train_loss=train_loss, 
                val_loss=val_loss,
                train_r2=train_r2, 
                val_r2=val_r2,
                epoch_time=time.perf_counter() - epoch_start,
                elapsed=time.perf_counter() - train_start)


def _layer_grad_norms(model: torch.nn.Module) -> list[float]:
    norms = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.grad is not None:
            norms.append(module.weight.grad.norm(2).item())
    return norms


def _r2_score(preds: torch.Tensor, targets: torch.Tensor) -> float:
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return 1.0 - (ss_res / ss_tot).item()