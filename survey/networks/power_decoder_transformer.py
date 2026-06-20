#!/usr/bin/env python3
"""Power et al. small decoder-only transformer."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('training')


from datasets import modular_equation_dataset
from experiment_core import ExperimentConfig, ExperimentSpec, print_summary, run_or_load


P = 97
OPERATION = "add"
SPLIT_SEED = 0
D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
DROPOUT = 0.0
LR = 1e-3
BATCH_SIZE = 512
WARMUP_STEPS = 10

CONFIG = ExperimentConfig(
    name="power_decoder_transformer",
    out_dir=Path("runs/power_decoder_transformer"),
    load_data=False,
    save_data=True,
    runs=1,
    seed=0,
    device="auto",
    num_threads=1,
    grok_train_frac=0.5,
    memorise_train_frac=0.5,
    grok_steps=20_000,
    memorise_steps=20_000,
    grok_weight_decay=1.0,
    memorise_weight_decay=0.0,
    eval_every=1000,
    log_every=100,
    n_checkpoints=25,
    include_initial_checkpoint=True,
    stop_early=False,
)

def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, bias=False, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.drop(a))
        return self.ln2(x + self.drop(self.ff(x)))


class PowerDecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, p: int, d_model: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pos", sinusoidal_positions(4, d_model), persistent=False)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos[: idx.size(1)].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x[:, -1, :])


def make_dataset(train_frac, device, config):
    return modular_equation_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return PowerDecoderTransformer(P + 2, P, D_MODEL, N_LAYERS, N_HEADS, DROPOUT)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=weight_decay)


def train_step(spec, model, optimizer, xtr, ytr, step, condition, device, config):
    lr_scale = min(1.0, step / max(1, WARMUP_STEPS))
    for group in optimizer.param_groups:
        group["lr"] = LR * lr_scale

    if BATCH_SIZE >= len(xtr):
        xb, yb = xtr, ytr
    else:
        idx = torch.randint(0, len(xtr), (BATCH_SIZE,), device=device)
        xb, yb = xtr[idx], ytr[idx]

    loss = spec.loss_fn(model, xb, yb, config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss


def metadata(config):
    return {
        "p": P,
        "operation": OPERATION,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
    }


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata, train_step=train_step)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
