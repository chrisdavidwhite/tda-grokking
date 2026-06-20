#!/usr/bin/env python3
"""Salah and Yevick modular-addition single-layer LSTM."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')

from datasets import modular_pair_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 53
OPERATION = "add"
SPLIT_SEED = 0
D_EMBED = 128
HIDDEN = 256
LR = 3e-4

CONFIG = ExperimentConfig(
    name="salah_single_layer_lstm",
    out_dir=Path("runs/salah_single_layer_lstm"),
    load_data=False,
    save_data=True,
    runs=5,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.5,
    memorise_train_frac=0.5,
    grok_steps=20_000,
    memorise_steps=20_000,
    grok_weight_decay=1.0,
    memorise_weight_decay=0.0,
    eval_every=500,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

class SalahLSTM(nn.Module):
    def __init__(self, p: int, d_embed: int, hidden: int):
        super().__init__()
        self.emb = nn.Embedding(p, d_embed)
        self.lstm = nn.LSTM(d_embed, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, p)
        self.reset()

    def reset(self):
        nn.init.xavier_normal_(self.emb.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.numel()
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        return self.fc(h[-1])


def make_dataset(train_frac, device, config):
    return modular_pair_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return SalahLSTM(P, D_EMBED, HIDDEN)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=weight_decay)


def metadata(config):
    return {"p": P, "operation": OPERATION, "d_embed": D_EMBED, "hidden": HIDDEN, "lr": LR}


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
