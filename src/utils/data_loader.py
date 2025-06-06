"""Utility functions to load datasets."""

from pathlib import Path
from typing import Tuple

import numpy as np
import mne


def load_physionet_dataset(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load PhysioNet Motor Imagery dataset.

    This function walks through ``data/raw/physionet`` located under
    ``path``, reads each EDF file with MNE and returns the EEG windows and
    their corresponding labels.

    Parameters
    ----------
    path : str or Path
        Root directory of the repository containing ``data/raw/physionet``.

    Returns
    -------
    X : np.ndarray
        Array of shape ``[n_trials, n_channels, n_times]`` containing EEG
        epochs.
    y : np.ndarray
        Array of integer labels associated to each trial. ``0 = rest``,
        ``1 = left`` and ``2 = right``.
    """
    data_dir = Path(path) / "data" / "raw" / "physionet"
    edf_files = sorted(data_dir.glob("*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found in {data_dir}")

    X_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    for file in edf_files:
        # Load raw EEG and set montage
        raw = mne.io.read_raw_edf(file, preload=True, verbose=False)
        raw.set_eeg_reference("average", projection=True)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, on_missing="ignore")

        # Drop problematic channels for topographic plots
        bad_chs = [
            "FCZ",
            "CZ",
            "CPZ",
            "FP1",
            "FPZ",
            "FP2",
            "AFZ",
            "FZ",
            "PZ",
            "POZ",
            "OZ",
            "IZ",
        ]
        raw.pick_channels([ch for ch in raw.ch_names if ch not in bad_chs])

        # Band-pass filter 1-40 Hz
        raw.filter(1.0, 40.0, fir_design="firwin", verbose=False)

        # Extract epochs based on annotations
        events, _ = mne.events_from_annotations(raw)
        event_id = dict(rest=1, left=2, right=3)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=0,
            tmax=2,
            baseline=None,
            preload=True,
            detrend=1,
            verbose=False,
        )

        if len(epochs) == 0:
            continue

        data = epochs.get_data()
        labels = epochs.events[:, 2]

        # Per-trial z-score
        data = (data - data.mean(axis=-1, keepdims=True)) / data.std(
            axis=-1, keepdims=True
        )

        X_all.append(data)
        y_all.append(labels)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)

    # Convert event ids (1,2,3) to 0-based labels
    y = y - 1

    return X, y
