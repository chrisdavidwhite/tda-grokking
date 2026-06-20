#!/usr/bin/env python3
"""MLP grokking experiment extracted from old1/tda_grokking.py."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_onehot_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


# ============================================================
# Defaults copied from old1/tda_grokking.py::Config.
# ============================================================

P = 43
OPERATION = "add"
HIDDEN1 = 256
HIDDEN2 = 128
LR = 1e-3
OPTIMIZER = "adamw"

CONFIG = ExperimentConfig(
    name="tda_grokking_mlp",
    out_dir=Path("runs/tda_grokking_mlp"),
    load_data=False,
    save_data=True,
    runs=25,
    seed=12321,
    device="auto",
    num_threads=1,
    grok_train_frac=0.40,
    memorise_train_frac=0.40,
    grok_steps=100_000,
    memorise_steps=100_000,
    grok_weight_decay=5.0,
    memorise_weight_decay=0.0,
    eval_every=200,
    log_every=200,
    eval_batch=8192,
    n_checkpoints=100,
    include_initial_checkpoint=True,
    checkpoint_style="log",
    log_checkpoint_min_step=1,
    stop_early=False,
)

class TDAGrokkingMLP(nn.Module):
    def __init__(self, p: int, hidden1: int, hidden2: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * p, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, p)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def old_seed(condition, run_idx, config):
    return config.seed + run_idx


def make_dataset(train_frac, device, config):
    split_seed = config.current_run_idx
    return modular_onehot_dataset(P, OPERATION, train_frac, split_seed, device)


def make_model(config):
    return TDAGrokkingMLP(P, HIDDEN1, HIDDEN2)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)


def metadata(config):
    return {
        "p": P,
        "operation": OPERATION,
        "hidden1": HIDDEN1,
        "hidden2": HIDDEN2,
        "optimizer": OPTIMIZER,
        "lr": LR,
        "source": "old1/tda_grokking.py",
    }

SPEC = ExperimentSpec(
    make_dataset=make_dataset,
    make_model=make_model,
    make_optimizer=make_optimizer,
    loss_fn=loss_fn,
    metadata=metadata,
    seed_fn=old_seed,
)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
