#!/usr/bin/env python
from __future__ import annotations

import contextlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DATASET_NAME = "experiment1"
MODEL_NAME = "wavenet_global_temp"
SCENARIO_NAME = "multi_domain_no_temperature_split70_15_15"
AUG_TAG = "no_aug"
KAGGLE_HISTORY_DIR = "/kaggle/working/History"
KAGGLE_WORKING_DATASET_ROOT = "/kaggle/working/DatasetPDT"
RAW_DATASET_ENV = "EXPERIMENT1_DATASET_ROOT"
RAW_ROOT_CANDIDATES = ["experiment1", "/kaggle/input/datasets/trtrnthanh/pasco-temp-dataset"]
TRAIN_VAL_TEST_DOMAIN = "experiment_30c"
EXTRA_TEST_DOMAINS = ["experiment_25c", "experiment_35c"]
EXPECTED_DOMAINS = [TRAIN_VAL_TEST_DOMAIN] + EXTRA_TEST_DOMAINS
EXPECTED_SIGNAL_CHANNELS = 4
TEMPERATURE_COLUMN_NAME = "temperature_c"
DOMAIN_TEMPERATURE_C = {
    "experiment_30c": 30.0,
    "experiment_25c": 25.0,
    "experiment_35c": 35.0,
}
ACC_SENSOR_KEYS = ["946-449", "964-462", "969-321", "966-489"]
CSV_SEP_CANDIDATES = [",", ";", None]
USE_TEMPERATURE_AUX = False
MULTI_TRAIN_DOMAINS = EXPECTED_DOMAINS


def normalize_col_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def resolve_dataset_root(candidate: str, expected_domains: list[str]) -> Path | None:
    root = Path(candidate)
    direct = root
    nested = root / "experiment1"
    for path in (direct, nested):
        if path.is_dir() and all((path / domain).is_dir() for domain in expected_domains):
            return path
    return None


def detect_raw_dataset_root(expected_domains: list[str]) -> str:
    candidates: list[str] = []
    env_root = os.environ.get(RAW_DATASET_ENV)
    if env_root:
        candidates.append(env_root)
    candidates.extend(RAW_ROOT_CANDIDATES)
    for candidate in candidates:
        root = resolve_dataset_root(candidate, expected_domains)
        if root is not None:
            return str(root)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for child in kaggle_input.iterdir():
            root = resolve_dataset_root(str(child), expected_domains)
            if root is not None:
                return str(root)
    raise FileNotFoundError(f"Cannot find raw dataset root containing domains: {expected_domains}")


def detect_npy_dataset_root(candidates: list[str], expected_domains: list[str]) -> str:
    for candidate in candidates:
        root = Path(candidate)
        if root.is_dir() and all((root / domain).is_dir() and any((root / domain).glob("*.npy")) for domain in expected_domains):
            return str(root)
    raise FileNotFoundError("Cannot find dataset root containing converted .npy files.")


def read_csv_flexible(file_path: Path) -> pd.DataFrame:
    last_error = None
    for sep in CSV_SEP_CANDIDATES:
        try:
            if sep is None:
                frame = pd.read_csv(file_path, sep=None, engine="python")
            else:
                frame = pd.read_csv(file_path, sep=sep)
            if frame.shape[1] > 1:
                return frame
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Failed to read CSV {file_path}: {last_error}")


def select_acc_columns(frame: pd.DataFrame, sensor_keys: list[str], expected_channels: int) -> list[str]:
    picked: list[str] = []
    for sensor_key in sensor_keys:
        for column in frame.columns:
            normalized = normalize_col_name(column)
            if "acceleration - z" in normalized and sensor_key in column and "run 1" in normalized:
                picked.append(column)
                break
    if len(picked) == expected_channels:
        return picked
    fallback = [column for column in frame.columns if "acceleration - z" in normalize_col_name(column) and "run 1" in normalize_col_name(column)]
    if len(fallback) < expected_channels:
        raise ValueError(f"Expected at least {expected_channels} acceleration-z columns, found {len(fallback)}. Columns={list(frame.columns)}")
    return fallback[:expected_channels]


def load_source_array(file_path: Path, domain: str) -> np.ndarray:
    frame = read_csv_flexible(file_path)
    columns = select_acc_columns(frame, ACC_SENSOR_KEYS, EXPECTED_SIGNAL_CHANNELS)
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    array = numeric.to_numpy(dtype=np.float32)
    if array.ndim != 2 or int(array.shape[1]) != EXPECTED_SIGNAL_CHANNELS:
        raise ValueError(f"Invalid array shape from {file_path}: {array.shape}")
    if not np.all(np.isfinite(array)):
        bad_rows = np.where(np.any(~np.isfinite(array), axis=1))[0]
        if bad_rows.size:
            first_bad = int(bad_rows[0])
            trailing_bad = np.arange(first_bad, array.shape[0])
            if np.array_equal(bad_rows, trailing_bad):
                trimmed = array[:first_bad, :]
                if trimmed.shape[0] > 0:
                    print(f"[TRIM] {file_path}: trimmed trailing bad rows from index {first_bad} to {array.shape[0] - 1}")
                    array = np.asarray(trimmed, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite values detected in selected columns of {file_path}")
    temperature = DOMAIN_TEMPERATURE_C[domain]
    temp_column = np.full((array.shape[0], 1), float(temperature), dtype=np.float32)
    return np.concatenate([np.asarray(array, dtype=np.float32), temp_column], axis=1)


def convert_source_to_npy(input_root: str, output_root: str, domains: list[str]) -> dict:
    in_root = Path(input_root)
    out_root = Path(output_root)
    converted = 0
    skipped = 0
    errors = 0
    for domain in domains:
        src_dir = in_root / domain
        out_dir = out_root / domain
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_files = sorted(path for path in src_dir.iterdir() if path.is_file() and path.suffix.lower() == ".csv")
        for source_path in tqdm(csv_files, desc=f"domain {domain}", leave=False):
            npy_path = out_dir / f"{source_path.stem}.npy"
            if npy_path.exists():
                skipped += 1
                continue
            try:
                array = load_source_array(source_path, domain)
                np.save(npy_path, np.asarray(array, dtype=np.float32))
                converted += 1
            except Exception as exc:
                errors += 1
                print(f"[ERROR] {source_path}: {exc}")
    return {"converted": converted, "skipped": skipped, "errors": errors}


def prepare_dataset_root() -> str:
    candidates = [KAGGLE_WORKING_DATASET_ROOT]
    try:
        dataset_root = detect_npy_dataset_root(candidates, EXPECTED_DOMAINS)
        print(f"Using existing npy dataset: {dataset_root}")
        return dataset_root
    except FileNotFoundError:
        raw_root = detect_raw_dataset_root(EXPECTED_DOMAINS)
        print(f"Found raw source dataset: {raw_root}")
        summary = convert_source_to_npy(raw_root, KAGGLE_WORKING_DATASET_ROOT, EXPECTED_DOMAINS)
        print("Conversion summary:", summary)
        dataset_root = detect_npy_dataset_root([KAGGLE_WORKING_DATASET_ROOT], EXPECTED_DOMAINS)
        print(f"Converted dataset ready at: {dataset_root}")
        return dataset_root


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_versioned_run_dir(base_dir: str, prefix: str = "v") -> tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    versions = [int(name[len(prefix):]) for name in os.listdir(base_dir) if name.startswith(prefix) and name[len(prefix):].isdigit()]
    version = f"{prefix}{(max(versions) + 1) if versions else 1}"
    run_dir = os.path.join(base_dir, version)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, version


def make_autocast(use_amp: bool, device_type: str):
    if not use_amp:
        return contextlib.nullcontext()
    try:
        from torch.amp import autocast as amp_autocast
        return amp_autocast(device_type=device_type, enabled=True)
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=True)


def make_scaler(use_amp: bool, device_type: str):
    if not use_amp:
        return None
    try:
        from torch.amp import GradScaler as amp_grad_scaler
        try:
            return amp_grad_scaler(device_type=device_type, enabled=True)
        except TypeError:
            return amp_grad_scaler(enabled=True)
    except Exception:
        from torch.cuda.amp import GradScaler as cuda_grad_scaler
        return cuda_grad_scaler(enabled=True)


def plot_performance(history: dict, save_base: str, title: str) -> None:
    del title
    xs = np.arange(1, len(history["loss"]) + 1)
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 15,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=200)
    ax1.plot(xs, np.asarray(history["acc"]) * 100.0, color="red", label="acc")
    ax1.plot(xs, np.asarray(history["val_acc"]) * 100.0, color="blue", label="val_acc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.grid(axis="x", color="0.85")
    ax1.grid(axis="y", color="0.85")
    ax1.legend(loc="best")
    ax2.plot(xs, history["loss"], color="red", label="loss")
    ax2.plot(xs, history["val_loss"], color="blue", label="val_loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(axis="x", color="0.85")
    ax2.grid(axis="y", color="0.85")
    ax2.legend(loc="best")
    fig.tight_layout(pad=0.8)
    fig.savefig(save_base + ".png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_probs: np.ndarray, save_base: str, title: str) -> None:
    del title
    cm = confusion_matrix(y_true, np.argmax(y_probs, axis=1))
    classes = [str(i) for i in range(cm.shape[0])]

    plt.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 24,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")

    ax.set_title("")
    ax.set_xlabel("Predicted label", fontsize=24, labelpad=12)
    ax.set_ylabel("True label", fontsize=24, labelpad=12)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("bottom")
    ax.tick_params(axis="x", labelrotation=0, pad=8)
    ax.tick_params(axis="y", pad=8)

    for text in ax.texts:
        text.set_fontsize(20)

    fig.tight_layout(pad=0.6)
    fig.savefig(save_base + ".png", dpi=500, bbox_inches="tight")
    plt.close(fig)


def build_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def starts_for_length(length: int, window: int) -> list[int]:
    if length < window:
        return []
    return list(range(0, length - window + 1, window))


@dataclass
class DataConfig:
    root: str
    domains: list[str]
    signal_channels: int
    window: int
    seed: int
    return_ct: bool
    use_temperature_aux: bool
    eps: float = 1e-8


def list_classes(root: str, domain: str) -> list[str]:
    domain_dir = Path(root) / domain
    classes = sorted(path.stem for path in domain_dir.glob("*.npy"))
    if not classes:
        raise FileNotFoundError(f"No .npy files found in {domain_dir}")
    return classes


def class_file_for_domain(cfg: DataConfig, domain: str, class_name: str) -> str:
    path = Path(cfg.root) / domain / f"{class_name}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing class file: {path}")
    return str(path)


def load_npy_mmap(path: str, cache: dict[str, np.ndarray], cfg: DataConfig) -> np.ndarray:
    if path not in cache:
        array = np.load(path, mmap_mode="r")
        expected_cols = int(cfg.signal_channels) + 1
        if array.ndim != 2 or int(array.shape[1]) != expected_cols:
            raise ValueError(f"Invalid array shape in {path}: {array.shape}")
        cache[path] = array
    return cache[path]


def load_window(path: str, start: int, cfg: DataConfig, cache: dict[str, np.ndarray]) -> np.ndarray:
    window = np.asarray(load_npy_mmap(path, cache, cfg)[start:start + cfg.window, :cfg.signal_channels], dtype=np.float32)
    if int(window.shape[0]) != int(cfg.window):
        raise ValueError(f"Window length mismatch in {path}: {window.shape[0]} != {cfg.window}")
    return window


def load_file_temperature(path: str, cfg: DataConfig, cache: dict[str, np.ndarray]) -> float:
    array = load_npy_mmap(path, cache, cfg)
    return float(array[0, cfg.signal_channels])


def build_index(files_with_label: list[tuple[str, int]], cfg: DataConfig) -> list[tuple[str, int, int]]:
    cache: dict[str, np.ndarray] = {}
    index: list[tuple[str, int, int]] = []
    for file_path, label in files_with_label:
        array = load_npy_mmap(file_path, cache, cfg)
        for start in starts_for_length(int(array.shape[0]), cfg.window):
            index.append((file_path, label, int(start)))
    if not index:
        raise ValueError("No windows were created from the dataset.")
    return index


def split_index_three_way_sequential(index: list[tuple[str, int, int]]) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    n_items = len(index)
    if n_items < 3:
        raise ValueError(f"Need at least 3 windows for sequential 70/15/15 split, got {n_items}.")
    train_end = int(n_items * 0.70)
    val_end = train_end + int(n_items * 0.15)
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(val_end, n_items - 1)
    train_index = index[:train_end]
    val_index = index[train_end:val_end]
    test_index = index[val_end:]
    if not train_index or not val_index or not test_index:
        raise ValueError(
            f"Sequential split produced empty partition: train={len(train_index)} val={len(val_index)} test={len(test_index)}"
        )
    return train_index, val_index, test_index


def build_index_three_way_per_file(files_with_label: list[tuple[str, int]], cfg: DataConfig) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    train_index: list[tuple[str, int, int]] = []
    val_index: list[tuple[str, int, int]] = []
    test_index: list[tuple[str, int, int]] = []
    for file_path, label in files_with_label:
        file_index = build_index([(file_path, label)], cfg)
        file_train, file_val, file_test = split_index_three_way_sequential(file_index)
        train_index.extend(file_train)
        val_index.extend(file_val)
        test_index.extend(file_test)
    if not train_index or not val_index or not test_index:
        raise ValueError(
            f"At least one split is empty after per-file sequential 70/15/15 partitioning: "
            f"train={len(train_index)} val={len(val_index)} test={len(test_index)}"
        )
    return train_index, val_index, test_index


def split_index_three_way_per_domain(classes: list[str], cfg: DataConfig) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    train_index: list[tuple[str, int, int]] = []
    val_index: list[tuple[str, int, int]] = []
    test_index: list[tuple[str, int, int]] = []
    for domain in cfg.domains:
        files_with_label = [
            (class_file_for_domain(cfg, domain, class_name), label)
            for label, class_name in enumerate(classes)
        ]
        domain_train, domain_val, domain_test = build_index_three_way_per_file(files_with_label, cfg)
        train_index.extend(domain_train)
        val_index.extend(domain_val)
        test_index.extend(domain_test)
    if not train_index or not val_index or not test_index:
        raise ValueError("At least one split is empty after per-file sequential 70/15/15 partitioning within each domain.")
    return train_index, val_index, test_index


def build_domain_only_split_index(files_with_label: list[tuple[str, int]], cfg: DataConfig, split: str) -> list[tuple[str, int, int]]:
    del split
    return build_index(files_with_label, cfg)


def preprocess_window(window: np.ndarray, eps: float) -> np.ndarray:
    processed = np.asarray(window, dtype=np.float32)
    processed = processed - processed.mean(axis=0, keepdims=True)
    local_std = processed.std(axis=0, keepdims=True)
    processed = processed / (local_std + eps)
    return processed


def temperature_stats_from_train(train_index: list[tuple[str, int, int]], cfg: DataConfig) -> tuple[float, float]:
    cache: dict[str, np.ndarray] = {}
    values = np.asarray([load_file_temperature(file_path, cfg, cache) for file_path, _, _ in train_index], dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    return mean, max(std, float(cfg.eps))


class WindowDataset(Dataset):
    def __init__(self, index: list[tuple[str, int, int]], cfg: DataConfig, temp_mean: float, temp_std: float):
        self.index = index
        self.cfg = cfg
        self.temp_mean = float(temp_mean)
        self.temp_std = float(temp_std)
        self.cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        file_path, label, start = self.index[idx]
        window = load_window(file_path, start, self.cfg, self.cache)
        temp = load_file_temperature(file_path, self.cfg, self.cache)
        window = preprocess_window(window, self.cfg.eps)
        temp = np.float32((temp - self.temp_mean) / self.temp_std)
        if self.cfg.return_ct:
            window = window.T
        return torch.from_numpy(window), torch.tensor([temp], dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def make_loaders(classes: list[str], cfg: DataConfig, batch_size: int, num_workers: int, loader_seed: int):
    train_index, val_index, test_index = split_index_three_way_per_domain(classes, cfg)
    temp_mean, temp_std = temperature_stats_from_train(train_index, cfg)

    train_ds = WindowDataset(train_index, cfg, temp_mean, temp_std)
    val_ds = WindowDataset(val_index, cfg, temp_mean, temp_std)
    test_ds = WindowDataset(test_index, cfg, temp_mean, temp_std)
    generator = torch.Generator().manual_seed(int(loader_seed))
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker if num_workers > 0 else None)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    meta = {
        "scenario": SCENARIO_NAME,
        "domains": cfg.domains,
        "train_samples": len(train_index),
        "val_samples": len(val_index),
        "test_samples": len(test_index),
        "class_files": {
            domain: {class_name: class_file_for_domain(cfg, domain, class_name) for class_name in classes}
            for domain in cfg.domains
        },
        "window_stride": int(cfg.window),
        "use_temperature_aux": cfg.use_temperature_aux,
        "domain_temperature_c": DOMAIN_TEMPERATURE_C,
    }
    return train_loader, val_loader, test_loader, meta, temp_mean, temp_std


def make_eval_loader(classes: list[str], cfg: DataConfig, batch_size: int, num_workers: int, temp_mean: float, temp_std: float):
    files_with_label = [
        (class_file_for_domain(cfg, domain, class_name), label)
        for domain in cfg.domains
        for label, class_name in enumerate(classes)
    ]
    eval_index = build_index(files_with_label, cfg)
    eval_ds = WindowDataset(eval_index, cfg, temp_mean, temp_std)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker if num_workers > 0 else None)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs)
    meta = {
        "domains": cfg.domains,
        "samples": len(eval_index),
        "class_files": {
            domain: {class_name: class_file_for_domain(cfg, domain, class_name) for class_name in classes}
            for domain in cfg.domains
        },
    }
    return eval_loader, meta


def make_domain_split_eval_loader(classes: list[str], cfg: DataConfig, batch_size: int, num_workers: int, temp_mean: float, temp_std: float, split: str):
    files_with_label = [
        (class_file_for_domain(cfg, domain, class_name), label)
        for domain in cfg.domains
        for label, class_name in enumerate(classes)
    ]
    eval_index = build_domain_only_split_index(files_with_label, cfg, split)
    eval_ds = WindowDataset(eval_index, cfg, temp_mean, temp_std)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker if num_workers > 0 else None)
    eval_loader = DataLoader(eval_ds, shuffle=False, **loader_kwargs)
    meta = {
        "domains": cfg.domains,
        "split": split,
        "samples": len(eval_index),
        "class_files": {
            domain: {class_name: class_file_for_domain(cfg, domain, class_name) for class_name in classes}
            for domain in cfg.domains
        },
    }
    return eval_loader, meta


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.dilation * (self.kernel_size - 1), 0)))


class WaveNetResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float, cond_dim: int, use_temperature_aux: bool):
        super().__init__()
        self.conv_tanh = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv_sig = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        self.use_temperature_aux = bool(use_temperature_aux)
        if self.use_temperature_aux:
            self.cond_tanh = nn.Linear(cond_dim, channels)
            self.cond_sig = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        tanh_term = self.conv_tanh(x)
        sig_term = self.conv_sig(x)
        if self.use_temperature_aux:
            if cond is None:
                raise ValueError("Conditioning tensor is required when temperature auxiliary input is enabled.")
            tanh_term = tanh_term + self.cond_tanh(cond).unsqueeze(-1)
            sig_term = sig_term + self.cond_sig(cond).unsqueeze(-1)
        gated = torch.tanh(tanh_term) * torch.sigmoid(sig_term)
        gated = self.dropout(gated)
        out = self.bn(self.conv_1x1(gated))
        return out + x, out


def wavenet_receptive_field(kernel_size: int, dilations: list[int], downsample_factor: int = 1) -> int:
    return int(downsample_factor) * (1 + (int(kernel_size) - 1) * sum(int(dilation) for dilation in dilations))


class WaveNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_channels: int,
        num_filters: int,
        kernel_size: int,
        number_of_blocks: int,
        residuals_per_block: int,
        downsample_factor: int,
        pool_type: str,
        dropout: float,
        use_temperature_aux: bool,
    ):
        super().__init__()
        self.use_temperature_aux = bool(use_temperature_aux)
        if downsample_factor > 1:
            if pool_type == "avg":
                self.pool = nn.AvgPool1d(downsample_factor, downsample_factor)
            elif pool_type == "max":
                self.pool = nn.MaxPool1d(downsample_factor, downsample_factor)
            else:
                raise ValueError("pool_type must be 'avg' or 'max'")
        else:
            self.pool = None
        self.dilations = [2 ** (i % residuals_per_block) for i in range(number_of_blocks * residuals_per_block)]
        self.receptive_field = wavenet_receptive_field(kernel_size, self.dilations, downsample_factor)
        if self.receptive_field > WINDOW:
            raise ValueError(f"WaveNet receptive field ({self.receptive_field}) must be <= WINDOW ({WINDOW}).")
        self.in_proj = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.cond_dim = num_filters
        if self.use_temperature_aux:
            self.temp_embed = nn.Sequential(
                nn.Linear(1, self.cond_dim),
                nn.ReLU(inplace=True),
            )
        self.blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(num_filters, kernel_size, dilation, dropout, self.cond_dim, self.use_temperature_aux)
                for dilation in self.dilations
            ]
        )
        self.head_conv1 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.head_bn1 = nn.BatchNorm1d(num_filters)
        self.head_conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=1)
        self.head_bn2 = nn.BatchNorm1d(num_filters)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x: torch.Tensor, temp: torch.Tensor | None = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        if self.pool is not None:
            x = self.pool(x)
        x = self.in_proj(x)
        cond = None
        if self.use_temperature_aux:
            if temp is None:
                raise ValueError("Temperature tensor is required when USE_TEMPERATURE_AUX=True.")
            cond = self.temp_embed(temp)
        skips = []
        for block in self.blocks:
            x, skip = block(x, cond)
            skips.append(skip)
        x = F.relu(torch.stack(skips, dim=0).sum(dim=0))
        x = F.relu(self.head_bn1(self.head_conv1(x)))
        x = F.relu(self.head_bn2(self.head_conv2(x)))
        return self.fc(self.gap(x).squeeze(-1))


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, n_class: int):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for xb, tb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb, tb if USE_TEMPERATURE_AUX else None)
            loss = criterion(logits, yb)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            total_loss += loss.item() * xb.size(0)
            y_true.append(yb.cpu().numpy())
            y_pred.append(preds.cpu().numpy())
            y_probs.append(probs.cpu().numpy())
    y_true = np.concatenate(y_true) if y_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred) if y_pred else np.array([], dtype=np.int64)
    y_probs = np.concatenate(y_probs) if y_probs else np.empty((0, n_class), dtype=np.float32)
    loss = total_loss / max(len(loader.dataset), 1)
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    return loss, acc, y_true, y_pred, y_probs


WINDOW = 1024
N_CHANNELS = EXPECTED_SIGNAL_CHANNELS
GLOBAL_SEED = 1
SPLIT_SEED = 1
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0
LABEL_SMOOTHING = 0.0
NUM_WORKERS = 0
USE_AMP = False
NUM_FILTERS = 64
KERNEL_SIZE = 2
NUMBER_OF_BLOCKS = 1
RESIDUALS_PER_BLOCK = 4
DOWNSAMPLE_FACTOR = 4
POOL_TYPE = "avg"
DROPOUT = 0.0


def main() -> None:
    os.makedirs(KAGGLE_HISTORY_DIR, exist_ok=True)
    dataset_root = prepare_dataset_root()
    classes = list_classes(dataset_root, MULTI_TRAIN_DOMAINS[0])
    cfg = DataConfig(
        root=dataset_root,
        domains=MULTI_TRAIN_DOMAINS,
        signal_channels=N_CHANNELS,
        window=WINDOW,
        seed=SPLIT_SEED,
        return_ct=False,
        use_temperature_aux=USE_TEMPERATURE_AUX,
    )
    set_global_determinism(GLOBAL_SEED)
    train_loader, val_loader, test_loader, meta, temp_mean, temp_std = make_loaders(classes, cfg, BATCH_SIZE, NUM_WORKERS, GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir, version = make_versioned_run_dir(KAGGLE_HISTORY_DIR)
    artifact_prefix = f"{DATASET_NAME}_{MODEL_NAME}_{SCENARIO_NAME}_{AUG_TAG}"
    checkpoint_path = os.path.join(run_dir, f"best_{artifact_prefix}.pt")
    model = WaveNetClassifier(
        num_classes=len(classes),
        input_channels=N_CHANNELS,
        num_filters=NUM_FILTERS,
        kernel_size=KERNEL_SIZE,
        number_of_blocks=NUMBER_OF_BLOCKS,
        residuals_per_block=RESIDUALS_PER_BLOCK,
        downsample_factor=DOWNSAMPLE_FACTOR,
        pool_type=POOL_TYPE,
        dropout=DROPOUT,
        use_temperature_aux=USE_TEMPERATURE_AUX,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    eval_criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=8)
    scaler = make_scaler(bool(USE_AMP and device.type == "cuda"), device.type)
    use_amp = bool(USE_AMP and device.type == "cuda" and scaler is not None)

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_epoch = -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for xb, tb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            tb = tb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with make_autocast(True, device.type):
                    logits = model(xb, tb if USE_TEMPERATURE_AUX else None)
                    loss = train_criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb, tb if USE_TEMPERATURE_AUX else None)
                loss = train_criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_seen += xb.size(0)
        train_loss = total_loss / max(total_seen, 1)
        train_acc = total_correct / max(total_seen, 1)
        val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, device, eval_criterion, len(classes))
        scheduler.step(val_loss)
        history["loss"].append(float(train_loss))
        history["acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        print(f"Epoch {epoch:03d}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_val_acc": best_val_acc,
                "config": {
                    "dataset_name": DATASET_NAME,
                    "model_name": MODEL_NAME,
                    "scenario_name": SCENARIO_NAME,
                    "aug_tag": AUG_TAG,
                    "data": asdict(cfg),
                    "training": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "label_smoothing": LABEL_SMOOTHING,
                        "num_workers": NUM_WORKERS,
                        "global_seed": GLOBAL_SEED,
                        "split_seed": SPLIT_SEED,
                        "local_centering": True,
                        "local_scaling": True,
                    },
                        "model": {
                            "num_filters": NUM_FILTERS,
                            "kernel_size": KERNEL_SIZE,
                            "number_of_blocks": NUMBER_OF_BLOCKS,
                            "residuals_per_block": RESIDUALS_PER_BLOCK,
                            "downsample_factor": DOWNSAMPLE_FACTOR,
                            "pool_type": POOL_TYPE,
                            "dropout": DROPOUT,
                            "use_temperature_aux": USE_TEMPERATURE_AUX,
                            "conditioning": "global",
                        },
                        "use_amp": USE_AMP,
                    },
                    "meta": meta,
                    "temperature_normalization": {"mean": temp_mean, "std": temp_std},
                    "classes": classes,
                }, checkpoint_path)

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    val_loss, val_acc, _, _, _ = evaluate_model(model, val_loader, device, eval_criterion, len(classes))
    test_loss, test_acc, test_true, test_pred, test_probs = evaluate_model(model, test_loader, device, eval_criterion, len(classes))
    history["best_epoch"] = int(payload["best_epoch"])
    history["best_val_acc"] = float(payload["best_val_acc"])
    history["best_val_loss"] = float(val_loss)
    history["test_loss"] = float(test_loss)
    history["test_acc"] = float(test_acc)
    history["run_dir"] = run_dir
    history["version"] = version

    scores = build_scores(test_true, test_pred)
    run_info = {
        "dataset_name": DATASET_NAME,
        "model_name": MODEL_NAME,
        "scenario_name": SCENARIO_NAME,
        "aug_tag": AUG_TAG,
        "version": version,
        "device": str(device),
        "classes": classes,
        "best_epoch": history["best_epoch"],
        "best_val_acc": history["best_val_acc"],
        "best_val_loss": history["best_val_loss"],
        "test_loss": history["test_loss"],
        "test_acc": history["test_acc"],
        "dataset_root": dataset_root,
        "config": payload["config"],
        "meta": meta,
    }
    with open(os.path.join(run_dir, f"history_{artifact_prefix}.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with open(os.path.join(run_dir, f"run_info_{artifact_prefix}.json"), "w", encoding="utf-8") as handle:
        json.dump(run_info, handle, indent=2)
    with open(os.path.join(run_dir, f"scores_{artifact_prefix}.json"), "w", encoding="utf-8") as handle:
        json.dump(scores, handle, indent=2)
    plot_performance(history, os.path.join(run_dir, f"learning_curve_{artifact_prefix}"), "")
    plot_confusion_matrix(test_true, test_probs, os.path.join(run_dir, f"confusion_matrix_{artifact_prefix}_test"), "")
    print(json.dumps({"run_dir": run_dir, "scores": scores}, indent=2))


if __name__ == "__main__":
    main()
