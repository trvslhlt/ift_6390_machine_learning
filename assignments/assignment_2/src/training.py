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
            optimizer.step()
            train_loss_sum += loss.item() * X.size(0)
            train_count += X.size(0)
            if on_batch_end:
                on_batch_end(
                    epoch=epoch, batch=batch_idx, loss=loss.item(), model=model,
                    batch_time=time.perf_counter() - batch_start,
                    elapsed=time.perf_counter() - train_start)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for X, y in validation_dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss_sum += loss.item() * X.size(0)
                val_count += X.size(0)

        train_loss = train_loss_sum / train_count
        val_loss = val_loss_sum / val_count
        if on_epoch_end:
            on_epoch_end(
                epoch=epoch, model=model, train_loss=train_loss, val_loss=val_loss,
                epoch_time=time.perf_counter() - epoch_start,
                elapsed=time.perf_counter() - train_start)