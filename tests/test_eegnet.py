import torch
from src.models.eegnet import EEGNetLight


def test_forward_output_shape():
    n_channels = 8
    n_timepoints = 128
    n_classes = 3
    model = EEGNetLight(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)
    x = torch.randn(2, 1, n_channels, n_timepoints)
    out = model(x)
    assert out.shape == (2, n_classes)
