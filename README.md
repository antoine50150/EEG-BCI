# 🧠 EEG-BCI Motor Imagery Classification with DeepConvNet

This project implements a complete Brain-Computer Interface (BCI) pipeline for motor imagery classification using EEG signals from the PhysioNet dataset. The goal is to build a robust, explainable, and scientifically rigorous deep learning model based on **DeepConvNet**, with custom enhancements such as:

- **Focal Loss** for class imbalance
- **Targeted Mixup** for minority classes
- **Temporal Dropout** to regularize time features
- **Light Attention** over feature maps
- Full **Stratified Cross-Validation** pipeline

---

## 📂 Project Structure

```bash
/notebooks/
├── 01_exploration_multi.ipynb         # EDF exploration and annotation checks
├── 02_preprocessing_deepconvnet.ipynb # Filtering, montage, epoching, normalization
├── 03_training_clean_deepconvnet.ipynb # DeepConvNet + Focal + Mixup + Attn
├── 04_analysis.ipynb                  # (to be added) Evaluation plots and per-class metrics
├── X_windows.npy                      # EEG input tensors [N, C, T]
├── y_windows.npy                      # Labels (rest, left, right)
```

---

## 🧪 Data

- Dataset: [PhysioNet Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/)
- Channels: 64 EEG (standard 10-20 montage)
- Classes: `0 = rest`, `1 = left hand`, `2 = right hand`
- Epoch length: 2s
- Filter: 1–40 Hz bandpass
- Z-score: Per-trial normalization

---

## 🚀 Model

**DeepConvNet** architecture with improvements:
- 4 convolutional blocks (25 → 200 filters)
- BatchNorm + ELU + MaxPooling
- Temporal Dropout + Simple Attention (optional modules)
- Linear classifier head

Training details:
- Optimizer: Adam
- Loss: Focal Loss (γ=2)
- Learning rate: `1e-4`
- Mixup α = 0.4 on classes 1 and 2
- Stratified 5-Fold Cross-Validation
- Epochs: 30 (modifiable)
- Batch size: 64

---

## 🧐 Results (preliminary)

| Fold | Accuracy | Macro F1 | Notes                      |
|------|----------|----------|----------------------------|
| 1    | 66.3%    | ~0.67    | Class 0 remains dominant   |
| 2    | 65.2%    | ~0.67    | Slight increase with mixup |
| 3    | TBD      |          |                            |
| 4    | TBD      |          |                            |
| 5    | TBD      |          |                            |

Next steps:
- Compare with baseline DeepConvNet
- Tune Focal Loss weights
- Add normalized confusion analysis
- Evaluate attention/dropout modules separately

---

## 🛠️ Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11
- mne
- scikit-learn
- numpy
- matplotlib
- seaborn

Install all requirements with Conda:
```bash
conda env create -f environment.yml
```

---

## 🗂️ Planned Features

- `04_analysis.ipynb`: class-wise F1, loss curves, confusion breakdown
- Optional: GUI demo or real-time decoding
- Future work: Riemannian pipelines, Transformer backbones

---

## 🤝 Contributing

This project is experimental and not open for contribution at the moment.

---

## License

This project is closed-source and not licensed for public use.  
**All rights reserved.**
