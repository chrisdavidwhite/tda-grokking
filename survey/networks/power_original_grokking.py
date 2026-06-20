#!/usr/bin/env python3
"""Power et al. original-style grokking transformer on modular arithmetic."""

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
OPERATION = "div"
SPLIT_SEED = 0

D_MODEL = 128
N_LAYERS = 2
N_HEADS = 4
D_HEAD = 32
D_MLP = 512
DROPOUT = 0.0
USE_BIAS = False
USE_LAYER_NORM = True

LR = 1e-3
BETA1 = 0.9
BETA2 = 0.98
BATCH_SIZE = 512
WARMUP_STEPS = 10
OPTIMIZER = "adam"

CONFIG = ExperimentConfig(
    name="power_original_grokking",
    out_dir=Path("runs/power_original_grokking"),
    load_data=False,
    save_data=True,
    runs=25,
    seed=0,
    device="auto",
    num_threads=1,
    same_condition_seeds=True,
    grok_train_frac=0.5,
    memorise_train_frac=0.5,
    grok_steps=1_000_000,
    memorise_steps=1_000_000,
    grok_weight_decay=0.0,
    memorise_weight_decay=0.0,
    eval_every=1000,
    log_every=100,
    n_checkpoints=100,
    include_initial_checkpoint=True,
    checkpoint_style="log",
    log_checkpoint_min_step=1,
    stop_early=False,
)


def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int, dropout: float, use_bias: bool, use_layer_norm: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, bias=use_bias, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.ln2 = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=use_bias),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=use_bias),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        tokens = x.size(1)
        mask = torch.triu(torch.ones(tokens, tokens, device=x.device, dtype=torch.bool), diagonal=1)
        attn, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.drop(attn))
        return self.ln2(x + self.drop(self.mlp(x)))


class OriginalGrokkingTransformer(nn.Module):
    def __init__(self, vocab_size: int, p: int):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, D_MODEL)
        self.register_buffer("position_embed", sinusoidal_positions(4, D_MODEL), persistent=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(D_MODEL, N_HEADS, D_HEAD, D_MLP, DROPOUT, USE_BIAS, USE_LAYER_NORM)
            for _ in range(N_LAYERS)
        ])
        self.unembed = nn.Linear(D_MODEL, p, bias=False)

    def forward(self, tokens):
        x = self.token_embed(tokens) + self.position_embed[: tokens.size(1)].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x[:, -1])


def make_dataset(train_frac, device, config):
    return modular_equation_dataset(P, OPERATION, train_frac, SPLIT_SEED, device)


def make_model(config):
    return OriginalGrokkingTransformer(P + 2, P)


def loss_fn(model, x, y, config):
    return F.cross_entropy(model(x), y)


def make_optimizer(model, weight_decay, condition, config):
    return torch.optim.Adam(model.parameters(), lr=LR, betas=(BETA1, BETA2), weight_decay=weight_decay)


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
        "architecture": "post_ln_sinusoidal_decoder",
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_head": D_HEAD,
        "d_mlp": D_MLP,
        "lr": LR,
        "optimizer": OPTIMIZER,
        "batch_size": BATCH_SIZE,
        "warmup_steps": WARMUP_STEPS,
        "use_layer_norm": USE_LAYER_NORM,
    }


SPEC = ExperimentSpec(make_dataset, make_model, make_optimizer, loss_fn, metadata, train_step=train_step)


def main():
    data = run_or_load(CONFIG, SPEC)
    print_summary(data)


if __name__ == "__main__":
    main()
