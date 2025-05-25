import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ajouter le dossier racine au path pour accéder à src/
sys.path.append(os.path.abspath(".."))
from src.models.eegnet import EEGNetLight

# Charger les données EEG déjà prétraitées
data = np.load("../../data/processed/subject_1_epochs.npz")
X = data["X"]  # (N, C, T)
y = data["y"]  # (N,)

# Préparation des tenseurs
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # [N, 1, C, T]
y_tensor = torch.tensor(y, dtype=torch.long)

# Dataset & DataLoader avec split entraînement/test
ds = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(ds))
test_size = len(ds) - train_size
train_ds, test_ds = random_split(ds, [train_size, test_size])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=8)

# Initialiser le modèle
n_channels, n_timepoints = X.shape[1], X.shape[2]
n_classes = len(np.unique(y))
model = EEGNetLight(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Suivi pour affichage
train_losses, train_accuracies = [], []

# Boucle d'entraînement
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
    print(f"🧠 Epoch {epoch+1:02d} — Loss: {total_loss:.4f} — Acc: {acc:.2f}%")

# Sauvegarde du modèle
model_path = "../../models/eegnet_subject1.pth"
torch.save(model.state_dict(), model_path)
print(f"\n✅ Modèle sauvegardé dans : {model_path}\n")

# Évaluation sur le jeu de test
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        preds = model(xb)
        y_true.extend(yb.numpy())
        y_pred.extend(preds.argmax(1).numpy())

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title("Matrice de confusion (test set)")
plt.show()

# Courbes loss & accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss au fil des epochs")
plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy au fil des epochs")
plt.tight_layout()
plt.show()
