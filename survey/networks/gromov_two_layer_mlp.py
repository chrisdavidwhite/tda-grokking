#!/usr/bin/env python3
"""Gromov two-layer MLP modular arithmetic experiment."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_onehot_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


# ============================================================
# Edit these variables directly.
# ============================================================

P = 97
OPERATION = "add"  # "add", "sub", or "mul"

WIDTH = 2048
ACTIVATION = "quadratic"  # "quadratic", "relu", or "tanh"
OUTPUT_SCALE = "mean_field"  # "mean_field", "sqrt_width", or "none"

LOSS = "mse_sum"  # "mse_sum", "mse_mean", or "ce"
OPTIMIZER = "sgd"  # "sgd" or "adamw"
LR = 5000.0
MOMENTUM = 0.0
BETA1 = 0.9
BETA2 = 0.98

CONFIG = ExperimentConfig(
    name="gromov_two_layer_mlp",
    out_dir=Path("runs/gromov_two_layer_mlp"),
    load_data=False,
    save_data=True,
    runs=25,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.5,
    memorise_train_frac=0.2,
    grok_steps=20_000,
    memorise_steps=20_000,
    grok_weight_decay=0.0,
    memorise_weight_decay=0.0,
    eval_every=1000,
    log_every=100,
    eval_batch=8192,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
    quick=False,
)

class GromovTwoLayerMLP(nn.Module):
    def __init__(self, p: int, width: int, activation: str, output_scale: str):
        super().__init__()
        self.p = p
        self.d_in = 2 * p
        self.width = width
        self.activation = activation
        self.output_scale = output_scale
        self.W1 = nn.Parameter(torch.randn(width, 2 * p))
        self.W2 = nn.Parameter(torch.randn(p, width))

    def phi(self, h):
        if self.activation == "quadratic":
            return h * h
        if self.activation == "relu":
            return F.relu(h)
        if self.activation == "tanh":
            return torch.tanh(h)
        raise ValueError(self.activation)

    def forward(self, x):
        h = x @ self.W1.t() / math.sqrt(self.d_in)
        out = self.phi(h) @ self.W2.t()
        if self.output_scale == "mean_field":
            return out / self.width
        if self.output_scale == "sqrt_width":
            return out / math.sqrt(self.width)
        if self.output_scale == "none":
            return out
        raise ValueError(self.output_scale)


def make_dataset(train_frac, device, config):
    return modular_onehot_dataset(P, OPERATION, train_frac, split_seed=0, device=device)


def make_model(config):
    return GromovTwoLayerMLP(P, WIDTH, ACTIVATION, OUTPUT_SCALE)


def classification_mse(pred, y):
    target = F.one_hot(y, P).to(dtype=pred.dtype)
    return 0.5 * ((pred - target) ** 2).sum(dim=-1).mean()


def loss_fn(model, x, y, config):
    pred = model(x)
    if LOSS == "mse_sum":
        return classification_mse(pred, y)
    if LOSS == "mse_mean":
        return F.mse_loss(pred, F.one_hot(y, P).to(dtype=pred.dtype))
    if LOSS == "ce":
        return F.cross_entropy(pred, y)
    raise ValueError(LOSS)


def make_optimizer(model, weight_decay, condition, config):
    lr = 1e-3 if OPTIMIZER == "adamw" and LR == 5000.0 else LR
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(BETA1, BETA2), weight_decay=weight_decay)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=weight_decay)


def metadata(config):
    return {
        "p": P,
        "operation": OPERATION,
        "width": WIDTH,
        "activation": ACTIVATION,
        "output_scale": OUTPUT_SCALE,
        "loss": LOSS,
        "optimizer": OPTIMIZER,
        "lr": LR,
    }


SPEC = ExperimentSpec(
    make_dataset=make_dataset,
    make_model=make_model,
    make_optimizer=make_optimizer,
    loss_fn=loss_fn,
    metadata=metadata,
)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
