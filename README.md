# WaveNet Global Conditioning for SHM Under Temperature Variation

This repository contains code and experiment outputs for structural damage detection using a WaveNet-based deep learning model, with a focus on improving robustness under temperature variation through global conditioning.

## Overview

The project studies damage classification from vibration and acceleration signals collected from a structure under multiple temperature conditions. The main comparison includes:

- a single-domain WaveNet baseline,
- a multi-domain WaveNet without temperature-aware conditioning,
- a multi-domain WaveNet with temperature used as a global conditioning variable.

The implementation uses:

- `PyTorch` for model training,
- `NumPy` and `Pandas` for data processing,
- `scikit-learn` for evaluation metrics,
- `Matplotlib` for learning curves and confusion matrices.

## Main Contribution

The main contribution of this repository is a WaveNet classifier that integrates temperature as an auxiliary global conditioning input. This is intended to improve the reliability of structural damage detection when the operating temperature changes.

In the current codebase:

- `train_single_domain.py` implements the single-domain baseline.
- `train_multi_domain_no_temperature.py` implements the multi-domain setting without temperature auxiliary input.
- `train_multi_domain_with_temperature.py` implements the multi-domain setting with temperature-based global conditioning.

## Experimental Settings

The repository currently contains three main experiment groups:

1. Single-domain training and evaluation
   - Training, validation, and testing are performed on the reference domain `experiment_30c`.
   - Additional testing is performed on `experiment_25c` and `experiment_35c`.

2. Multi-domain training without temperature auxiliary input
   - Training samples are drawn from multiple temperature domains.
   - The model does not explicitly use temperature as a conditioning variable.

3. Multi-domain training with temperature auxiliary input
   - Training samples are drawn from multiple temperature domains.
   - Temperature is normalized and injected into the WaveNet residual blocks as a global conditioning signal.

The domain-temperature mapping used in the current scripts is:

- `experiment_30c` -> `30 C`
- `experiment_25c` -> `25 C`
- `experiment_35c` -> `35 C`

## Damage Classes

The current experiments use 8 classes:

- `no_damage`
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
|-- train_single_domain.py
|-- train_multi_domain_no_temperature.py
|-- train_multi_domain_with_temperature.py
|-- single_domain_train25/
|-- single_domain_train30/
|-- single_domain_train35/
|-- multi_domain_no_temperature/
`-- multi_domain_with_temperature/
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
experiment1/
|-- experiment_30c/
|-- experiment_25c/
`-- experiment_35c/
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

Install the main dependencies with:

```bash
pip install torch numpy pandas matplotlib scikit-learn tqdm
```

## How To Run

### 1. Single-domain baseline

```bash
python train_single_domain.py
```

### 2. Multi-domain without temperature auxiliary input

```bash
python train_multi_domain_no_temperature.py
```

### 3. Multi-domain with temperature global conditioning

```bash
python train_multi_domain_with_temperature.py
```

## Dataset Path Configuration

The scripts support dataset root detection through:

- the environment variable `EXPERIMENT1_DATASET_ROOT`, or
- built-in local or Kaggle candidate paths.

If needed, set the dataset root manually before running:

```bash
set EXPERIMENT1_DATASET_ROOT=path\to\experiment1
python train_multi_domain_with_temperature.py
```

On PowerShell:

```powershell
$env:EXPERIMENT1_DATASET_ROOT="path\to\experiment1"
python train_multi_domain_with_temperature.py
```

## Model and Training Details

The current scripts use a WaveNet-style architecture with:

- causal convolutions,
- dilated residual blocks,
- skip connections,
- global average pooling,
- a classification head for damage-state prediction.

The current default settings include:

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
- the serialized best checkpoint.

These outputs support result tracking and direct reuse in figures or tables for publications.
