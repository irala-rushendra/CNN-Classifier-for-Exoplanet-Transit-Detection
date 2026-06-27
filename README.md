# TESS Exoplanet Classifier

A 1D dual-branch CNN that classifies TESS light curves into 6 disposition categories: Planet Candidate (PC), False Positive (FP), Confirmed Planet (CP), Kepler Candidate (KP), Astrophysical Candidate (APC), and False Alarm (FA).
(This is still undergoing project)
## Architecture

- **Global view branch**: processes the full folded light curve (2001 time steps) through 3 conv blocks
- **Local view branch**: processes a zoomed-in transit window (201 time steps) through 2 conv blocks
- **Merged head**: concatenated features → 3 fully-connected layers with dropout

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### 1. Create catalog (CSV → labeled catalog)

```bash
python create_catalog.py
```

### 2. Download light curves and build dataset

```bash
python build_dataset.py
```

Supports checkpoint resume. Use `--no-resume` to start fresh:
```bash
python build_dataset.py --no-resume
```

### 3. Train

```bash
python train.py
```

Configurable options:
```bash
python train.py --dataset data/processed/training_dataset.pt --epochs 100 --lr 5e-5
```

### 4. Evaluate

```bash
python evaluate.py
```

### 5. Run inference (synthetic demo)

```bash
python predict.py
```

### Custom paths

All scripts accept `--dataset`, `--weights`, and `--batch-size` arguments where applicable.

## Project structure

```
├── build_dataset.py       # Download TESS light curves & save as .pt
├── create_catalog.py      # Build labeled catalog from raw CSV
├── train.py               # Train the model
├── evaluate.py            # Evaluate on test set
├── predict.py             # Inference with synthetic data
├── src/
│   ├── model.py           # ExoplanetDualBranchCNN definition
│   └── utils.py           # Shared constants and helpers
├── data/
│   ├── raw/               # Raw CSV and .pt files
│   └── processed/         # Catalogs and training datasets
└── weights/               # Trained model weights
```
