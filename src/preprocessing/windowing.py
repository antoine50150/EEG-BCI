import mne
import numpy as np
import os
import builtins
print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)


def extract_epochs_from_annotations(raw: mne.io.Raw, tmin: float, tmax: float, output_path: str = None):
    """
    Découpe le signal EEG autour des annotations en fenêtres (epochs).

    Args:
        raw: Objet Raw MNE déjà prétraité (filtré, ICA, etc.)
        tmin: Temps avant l’annotation (en secondes)
        tmax: Temps après l’annotation (en secondes)
        output_path: Chemin pour sauvegarder le fichier .npz (facultatif)

    Returns:
        X: np.ndarray (n_epochs, n_channels, n_times)
        y: np.ndarray (n_epochs,)
    """
    print("\u2705 Extraction des événements depuis les annotations...")
    events, event_id = mne.events_from_annotations(raw)

    print("✅ Extraction des événements réussie.")


    print(f"\u2705 Création des epochs : tmin = {tmin}s, tmax = {tmax}s")
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=tmin, tmax=tmax,
        baseline=None,
        detrend=1,
        preload=True
    )

    print("\u2705 Extraction des données numpy...")
    X = epochs.get_data()                # Shape: (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]            # Dernière colonne : id de l'événement

    if output_path:
        np.savez(output_path, X=X, y=y)
        print(f"\ud83d\udcc4 Données sauvegardées dans : {output_path}")

    return X, y
