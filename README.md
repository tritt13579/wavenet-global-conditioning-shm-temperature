# WaveNet Global Conditioning for SHM Under Temperature Variation

This repository contains the code and experiment outputs for structural damage detection using a WaveNet-based deep learning model, with a focus on improving robustness under temperature variation through global conditioning.

## Overview

The project studies damage classification from vibration/acceleration signals collected from a structure under multiple temperature conditions. The main idea is to compare:

- a single-domain WaveNet baseline,
- a multi-domain WaveNet without temperature-aware conditioning,
- a multi-domain WaveNet with temperature used as a global conditioning variable.

The current implementation uses:

- `PyTorch` for model training,
- `NumPy` and `Pandas` for data processing,
- `scikit-learn` for evaluation metrics,
- `Matplotlib` for learning curves and confusion matrices.

## Main Contribution

The central contribution of this repository is a WaveNet classifier that integrates temperature as an auxiliary global conditioning input. This is designed to improve the reliability of structural damage detection when the operating temperature changes.

In the current codebase:

- `train_kichban_donmien.py` implements the single-domain baseline.
- `train_kichban_khongnhiet.py` implements the multi-domain setting without temperature auxiliary input.
- `train_kichban3_conhiet.py` implements the multi-domain setting with temperature-based global conditioning.

## Experimental Settings

The repository currently contains three main experiment groups:

1. Single-domain training and evaluation
   - Training, validation, and testing are performed on the reference domain `thinghiem_L0`.
   - Additional testing is performed on `thinghiem_L0minusdelta` and `thinghiem_L0plusdelta`.

2. Multi-domain training without temperature auxiliary input
   - Training samples are drawn from multiple temperature domains.
   - The model does not explicitly use temperature as a conditioning variable.

3. Multi-domain training with temperature auxiliary input
   - Training samples are drawn from multiple temperature domains.
   - Temperature is normalized and injected into the WaveNet residual blocks as a global conditioning signal.

The domain-temperature mapping used in the current scripts is:

- `thinghiem_L0` -> `30 C`
- `thinghiem_L0minusdelta` -> `25 C`
- `thinghiem_L0plusdelta` -> `35 C`

## Damage Classes

The current experiments use 8 classes:

- `khong_m`
- `m1`
- `m1m2`
- `m1m2m3`
- `m1m3`
- `m2`
- `m2m3`
- `m3`

These labels are read directly from the dataset files.

## Repository Structure

```text
.
├── train_kichban_donmien.py
├── train_kichban_khongnhiet.py
├── train_kichban3_conhiet.py
├── kichban_donmien_train25/
├── kichban_donmien_train30/
├── kichban_donmien_train35/
├── kichban_damien_khongnhiet/
└── kichban_damien_conhiet/
```

The result folders contain experiment artifacts such as:

- `history_*.json`
- `run_info_*.json`
- `scores_*.json`
- `extra_test_scores_*.json`
- `learning_curve_*.png`
- `confusion_matrix_*.png`
- `combined_confusion_matrix_*.png`

## Data Format

The scripts expect input data organized by domain, for example:

```text
thinghiem1/
├── thinghiem_L0/
├── thinghiem_L0minusdelta/
└── thinghiem_L0plusdelta/
```

Each domain folder contains source files for the damage classes. The scripts:

- read CSV files,
- select 4 acceleration-z channels,
- convert them to `.npy`,
- build fixed windows for training and evaluation.

For the temperature-aware setting, a temperature column is appended during preprocessing and later used as an auxiliary variable.

## Requirements

Recommended environment:

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

You can install the main dependencies with:

```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

## How To Run

### 1. Single-domain baseline

```bash
python train_kichban_donmien.py
```

### 2. Multi-domain without temperature auxiliary input

```bash
python train_kichban_khongnhiet.py
```

### 3. Multi-domain with temperature global conditioning

```bash
python train_kichban3_conhiet.py
```

## Dataset Path Configuration

The scripts support dataset root detection through:

- the environment variable `THINGHIEM1_DATASET_ROOT`, or
- built-in local/Kaggle candidate paths.

If needed, set the dataset root manually before running:

```bash
set THINGHIEM1_DATASET_ROOT=path\to\thinghiem1
python train_kichban3_conhiet.py
```

On PowerShell:

```powershell
$env:THINGHIEM1_DATASET_ROOT="path\to\thinghiem1"
python train_kichban3_conhiet.py
```

## Model and Training Details

The current scripts use a WaveNet-style architecture with:

- causal convolutions,
- dilated residual blocks,
- skip connections,
- global average pooling,
- classification head for damage state prediction.

The current default settings in the scripts include:

- window size: `1024`
- batch size: `32`
- epochs: `100`
- learning rate: `1e-3`
- optimizer: `Adam`
- scheduler: `ReduceLROnPlateau`

For the temperature-aware experiment, temperature is:

- extracted from the domain definition,
- normalized using training statistics,
- fed to the model as a global auxiliary conditioning variable.

## Outputs

Each run saves:

- training history,
- validation and test metrics,
- confusion matrices,
- learning curves,
- run configuration metadata,
- serialized best checkpoint.

These outputs support result tracking and direct use in figures or tables for publications.

