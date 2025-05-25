import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetLight(nn.Module):
    def __init__(self, n_channels, n_timepoints, n_classes):
        super().__init__()

        self.temporal = nn.Conv2d(1, 8, kernel_size=(1, 64), padding=(0, 32))
        self.spatial = nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AvgPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(0.5)

        # Calcul dynamique de la taille d'entrée de la couche linéaire
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            x = self.temporal(dummy)
            x = self.spatial(x)
            x = self.bn(x)
            x = self.pool(x)
            x = self.dropout(x)
            self.flattened_size = x.numel()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, n_classes)
        )

    def forward(self, x):
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
