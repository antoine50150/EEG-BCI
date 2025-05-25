import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mne

# Ajouter le dossier racine au path pour accéder à src/
sys.path.append(os.path.abspath(".."))
from src.models.eegnet import EEGNetLight

def extract_epochs_from_annotations(raw, tmin=0.0, tmax=4.0, output_path=None):
    print("✅ Extraction des événements depuis les annotations...")
    descriptions = list(set(annot["description"] for annot in raw.annotations))
    descriptions.sort()
    event_id = {desc: idx for idx, desc in enumerate(descriptions)}

    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    print(f"✅ {len(events)} événements détectés : {event_id}")

    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None,
                        preload=True, detrend=1, verbose=False)
    X = epochs.get_data()  # (n_epochs, n_channels, n_timepoints)
    y = epochs.events[:, -1]

    if output_path:
        np.savez(output_path, X=X, y=y)
        print(f"📀 Sauvegarde effectuée dans : {output_path}")

    print(f"🔢 Extraction des données numpy : {X.shape} — Labels : {np.unique(y)}")
    return X, y

def train_and_evaluate():
    data = np.load("../../data/processed/subject_1_epochs.npz")
    X = data["X"]
    y = data["y"]
    y -= y.min()

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    ds = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.8 * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = random_split(ds, [train_size, test_size])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8)

    n_channels, n_timepoints = X.shape[1], X.shape[2]
    n_classes = len(torch.unique(y_tensor))
    model = EEGNetLight(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    train_losses, train_accuracies = [], []

    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (preds.argmax(1) == yb).sum().item()
            total += yb.size(0)

        acc = 100 * correct / total
        train_losses.append(total_loss)
        train_accuracies.append(acc)
        print(f"🤔 Epoch {epoch+1:02d} — Loss: {total_loss:.4f} — Acc: {acc:.2f}%")

    model_path = "../../models/eegnet_subject1.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Modèle sauvegardé dans : {model_path}\n")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb)
            y_true.extend(yb.numpy())
            y_pred.extend(preds.argmax(1).numpy())

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Matrice de confusion (test set)")
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss au fil des epochs")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Accuracy", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy au fil des epochs")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
