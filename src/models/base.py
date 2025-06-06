from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module, ABC):
    """Abstract base model class with common training utilities."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that must be implemented by subclasses."""
        raise NotImplementedError

    def training_step(self, batch):
        """Single training step returning the loss."""
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        """Single validation step returning loss and accuracy."""
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, targets)
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean()
        return {"val_loss": loss, "val_acc": acc}
