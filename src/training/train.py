import os
from typing import List, Dict

import torch
from torch.utils.data import DataLoader


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
) -> List[Dict[str, float]]:
    """Train a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    train_loader : DataLoader
        Dataloader for training data.
    val_loader : DataLoader
        Dataloader for validation data.
    epochs : int, optional
        Number of epochs. Default is ``10``.
    lr : float, optional
        Learning rate. Default is ``1e-3``.
    checkpoint_dir : str, optional
        Directory to save checkpoints. Default is ``"checkpoints"``.

    Returns
    -------
    List[Dict[str, float]]
        Metrics for each epoch (train loss, val loss, val accuracy).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(checkpoint_dir, exist_ok=True)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_running_loss += loss.item() * xb.size(0)
                correct += (preds.argmax(1) == yb).sum().item()
        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = 100 * correct / len(val_loader.dataset)

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }, checkpoint_path)

    return history
