#!/usr/bin/env python3
"""Offline persistent-homology analysis for one saved experiment run."""

from __future__ import annotations

import importlib
import math
import os
import pickle
import re
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ripser import ripser
import sys
sys.path.append('training')
sys.path.append('networks')


# ============================================================
# Edit these variables directly.
# ============================================================

# EXPERIMENT_MODULE = "power_original_grokking"
# EXPERIMENT_MODULE = "gromov_two_layer_mlp"
#EXPERIMENT_MODULE = "salah_embedding_concat_mlp"
EXPERIMENT_MODULE = "tda_grokking_mlp"
RUN_SERIAL = 1 # 1, "001", etc.; None means newest serial in the experiment folder.
DATA_PATH = None  # Optional exact pickle path. Overrides RUN_SERIAL.

RUN_INDEX = "all"  # 0, 1, 2, ... or "all"
MAX_RUNS_PER_CONDITION = None # None means all; otherwise load this many grok and memorise runs.

POINT_CLOUD = "weight_rows"  # "activations" or "weight_rows"
LAYER_NAME = "fc2"  # Examples: "mlp.0", "attn.o", "layers.0.fc1", "fc1", "W2"
INCLUDE_TDA_SYMMETRY = True  # Adds a swap-symmetry panel for tda_grokking_mlp.
TDA_SYMMETRY_LAYER = "both"  # "fc1", "fc2", or "both".
LOAD_PERSISTENCE_DIAGRAMS = False  # If True, load cached diagrams and skip Ripser when the .pkl exists.
PERSISTENCE_DIAGRAMS_PATH = None  # Optional exact cache path. None saves under the figure output directory.

HOMOLOGY_DIMS = (1,2)  # Use (1, 2) for H1 and H2.
MAX_POINTS = None  # Set to an int if Ripser is too slow.
STANDARDIZE_POINTS = True

ACTIVATION_BATCH_SIZE = 1024
BETTI_EPS_MODE = "auto"  # "auto", "relative", or "absolute"
BETTI_EPS = 0.75  # relative: multiplier of median pairwise distance; absolute: direct radius
TOTAL_PERSISTENCE_POWER = 1.0
X_AXIS_LOG10 = True
CURVE_STYLE = "mean_std"  # "mean_all" or "mean_std"
GROKKING_RANGE_START_FRAC = 0.10
GROKKING_RANGE_END_FRAC = 0.90

OUT_DIR = Path("figures")
CONDITION_COLORS = {"grok": "#1d4ed8", "memorise": "#dc2626"}
TDA_SYMMETRY_EPS = 1e-12


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def serial_from_path(path: Path) -> int | None:
    match = re.match(r"^(\d{3})_", path.name)
    return int(match.group(1)) if match else None


def formatted_serial(serial) -> str:
    return f"{int(serial):03d}"


def serial_pickle_files(out_dir: Path) -> list[tuple[int, Path]]:
    files = []
    for path in out_dir.glob("*.pkl"):
        serial = serial_from_path(path)
        if serial is not None:
            files.append((serial, path))
    return sorted(files, key=lambda item: (item[0], item[1].stat().st_mtime))


def resolve_data_path(module) -> Path:
    if DATA_PATH is not None:
        path = Path(DATA_PATH)
        if path.exists():
            return path
        raise FileNotFoundError(f"DATA_PATH does not exist: {path}")

    serial_files = serial_pickle_files(module.CONFIG.out_dir)
    if RUN_SERIAL is not None:
        serial = int(RUN_SERIAL)
        matches = [path for file_serial, path in serial_files if file_serial == serial]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            candidate_lines = "\n".join(f"  - {path}" for path in matches)
            raise FileNotFoundError(
                f"Serial {formatted_serial(RUN_SERIAL)} is ambiguous in {module.CONFIG.out_dir}:\n"
                f"{candidate_lines}"
            )
        candidate_lines = "\n".join(f"  - {path}" for _, path in serial_files) or "  (none)"
        raise FileNotFoundError(
            f"Could not find serial {formatted_serial(RUN_SERIAL)} in {module.CONFIG.out_dir}.\n"
            f"Available serialised .pkl files:\n{candidate_lines}"
        )

    if serial_files:
        return serial_files[-1][1]

    fallback = module.CONFIG.data_path
    if fallback.exists():
        return fallback

    candidates = sorted(module.CONFIG.out_dir.glob("*.pkl"))
    candidate_lines = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
    raise FileNotFoundError(
        "Could not find saved training data for the current experiment config.\n"
        f"Looked for serialised files in:\n  - {module.CONFIG.out_dir}\n"
        f"Legacy fallback:\n  - {fallback}\n"
        f"Available .pkl files in {module.CONFIG.out_dir}:\n{candidate_lines}\n\n"
        "Run the experiment first, set RUN_SERIAL, or set DATA_PATH in topology_analysis.py."
    )


def validate_settings():
    if CURVE_STYLE not in {"mean_all", "mean_std"}:
        raise ValueError('CURVE_STYLE must be "mean_all" or "mean_std"')
    if MAX_RUNS_PER_CONDITION is not None and MAX_RUNS_PER_CONDITION < 1:
        raise ValueError("MAX_RUNS_PER_CONDITION must be None or a positive integer")
    if TDA_SYMMETRY_LAYER not in {"fc1", "fc2", "both"}:
        raise ValueError('TDA_SYMMETRY_LAYER must be "fc1", "fc2", or "both"')


def warn_if_data_mismatch(data, module, data_path):
    meta = data.get("meta", {})
    expected = module.SPEC.metadata(module.CONFIG)
    mismatches = []
    for key, expected_value in expected.items():
        if key in meta and meta[key] != expected_value:
            mismatches.append((key, meta[key], expected_value))

    run_mismatches = []
    for key in ("runs", "n_checkpoints"):
        expected_value = getattr(module.CONFIG, key, None)
        if key in meta and expected_value is not None and meta[key] != expected_value:
            run_mismatches.append((key, meta[key], expected_value))

    if mismatches or run_mismatches:
        print(f"WARNING: loaded data does not match current {EXPERIMENT_MODULE} config: {data_path}", flush=True)
        for key, loaded_value, expected_value in mismatches + run_mismatches:
            print(f"  {key}: loaded={loaded_value!r}, current={expected_value!r}", flush=True)


def selected_runs(data):
    conditions = ["grok", "memorise"]
    if RUN_INDEX == "all":
        matches = [row for row in data["results"] if row["condition"] in conditions]
    else:
        matches = [
            row for row in data["results"]
            if row["condition"] in conditions and row["run"] == RUN_INDEX
        ]
    if not matches:
        raise ValueError(f"Could not find grok/memorise run {RUN_INDEX}")
    found_conditions = {row["condition"] for row in matches}
    missing_conditions = sorted(set(conditions) - found_conditions)
    if missing_conditions:
        raise ValueError(
            f"Expected both grok and memorise runs, but missing: {', '.join(missing_conditions)}"
        )

    matches = sorted(matches, key=lambda row: (row["condition"], row["run"]))
    if MAX_RUNS_PER_CONDITION is None:
        return matches

    limited = []
    for condition in conditions:
        condition_runs = [row for row in matches if row["condition"] == condition]
        limited.extend(condition_runs[:MAX_RUNS_PER_CONDITION])
    return sorted(limited, key=lambda row: (row["condition"], row["run"]))


def run_label(run):
    return f"{run['condition']}_run{run['run']}"


def output_serial_label(data_path: Path) -> str:
    serial = serial_from_path(data_path)
    return f"{serial:03d}" if serial is not None else "no_serial"


def base_out_dir(data_path: Path):
    return OUT_DIR / EXPERIMENT_MODULE / output_serial_label(data_path) / POINT_CLOUD / LAYER_NAME.replace(".", "_")


def persistence_diagrams_path(base_dir: Path) -> Path:
    if PERSISTENCE_DIAGRAMS_PATH is not None:
        return Path(PERSISTENCE_DIAGRAMS_PATH)
    return base_dir / "persistence_diagrams.pkl"


def persistence_cache_settings(data_path: Path) -> dict:
    return {
        "experiment_module": EXPERIMENT_MODULE,
        "data_path": str(data_path),
        "run_serial": RUN_SERIAL,
        "run_index": RUN_INDEX,
        "max_runs_per_condition": MAX_RUNS_PER_CONDITION,
        "point_cloud": POINT_CLOUD,
        "layer_name": LAYER_NAME,
        "include_tda_symmetry": INCLUDE_TDA_SYMMETRY,
        "tda_symmetry_layer": TDA_SYMMETRY_LAYER,
        "homology_dims": tuple(HOMOLOGY_DIMS),
        "max_points": MAX_POINTS,
        "standardize_points": STANDARDIZE_POINTS,
        "betti_eps_mode": BETTI_EPS_MODE,
        "betti_eps": BETTI_EPS,
        "total_persistence_power": TOTAL_PERSISTENCE_POWER,
    }


def load_persistence_diagrams(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "run_tdas" in payload:
        return payload["run_tdas"]
    return payload


def save_persistence_diagrams(path: Path, data_path: Path, runs, run_tdas):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "settings": persistence_cache_settings(data_path),
        "run_labels": [run_label(run) for run in runs],
        "run_tdas": run_tdas,
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved persistence diagrams to {path}", flush=True)


def get_module(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        print("Available module names:")
        for module_name in modules:
            print(f"  {module_name}")
        raise ValueError(f"Unknown module: {name}")
    return modules[name]


class LegacyCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, use_bias: bool):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        inner = n_heads * d_head
        self.q = nn.Linear(d_model, inner, bias=use_bias)
        self.k = nn.Linear(d_model, inner, bias=use_bias)
        self.v = nn.Linear(d_model, inner, bias=use_bias)
        self.o = nn.Linear(inner, d_model, bias=use_bias)

    def forward(self, x):
        batch, tokens, _ = x.shape

        def split(z):
            return z.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)

        q = split(self.q(x))
        k = split(self.k(x))
        v = split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(tokens, tokens, device=x.device, dtype=torch.bool), diagonal=1)
        attn = torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        y = (attn @ v).transpose(1, 2).contiguous().view(batch, tokens, self.n_heads * self.d_head)
        return self.o(y)


class LegacyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, d_mlp: int, use_bias: bool):
        super().__init__()
        self.attn = LegacyCausalAttention(d_model, n_heads, d_head, use_bias)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp, bias=use_bias),
            nn.ReLU(),
            nn.Linear(d_mlp, d_model, bias=use_bias),
        )

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class LegacyOriginalGrokkingTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        p: int,
        seq_len: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        d_mlp: int,
        use_bias: bool,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.register_buffer("position_embed", torch.zeros(seq_len, d_model))
        self.blocks = nn.ModuleList([
            LegacyTransformerBlock(d_model, n_heads, d_head, d_mlp, use_bias)
            for _ in range(n_layers)
        ])
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, tokens):
        x = self.token_embed(tokens) + self.position_embed[: tokens.size(1)].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x[:, -1])


def is_legacy_attention_state_dict(state_dict):
    return any(re.fullmatch(r"blocks\.\d+\.attn\.q\.weight", key) for key in state_dict)


def make_legacy_model_from_state_dict(state_dict):
    token_weight = state_dict["token_embed.weight"]
    unembed_weight = state_dict["unembed.weight"]
    d_model = token_weight.shape[1]
    vocab_size = token_weight.shape[0]
    p = unembed_weight.shape[0]
    seq_len = state_dict.get("position_embed", torch.empty(4, d_model)).shape[0]
    layer_ids = {
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(r"blocks\.(\d+)\..+", key))
    }
    n_layers = max(layer_ids) + 1
    n_heads = getattr(importlib.import_module(EXPERIMENT_MODULE), "N_HEADS", 4)
    inner = state_dict["blocks.0.attn.q.weight"].shape[0]
    d_head = inner // n_heads
    d_mlp = state_dict["blocks.0.mlp.0.weight"].shape[0]
    use_bias = "blocks.0.attn.q.bias" in state_dict or "blocks.0.mlp.0.bias" in state_dict
    return LegacyOriginalGrokkingTransformer(
        vocab_size, p, seq_len, d_model, n_layers, n_heads, d_head, d_mlp, use_bias
    )


def load_checkpoint_model(module, checkpoint):
    if "state_dict" not in checkpoint:
        raise ValueError(
            "This checkpoint has no state_dict. Rerun training with the updated experiment_core.py."
        )
    state_dict = checkpoint["state_dict"]
    if is_legacy_attention_state_dict(state_dict):
        model = make_legacy_model_from_state_dict(state_dict)
    else:
        model = module.SPEC.make_model(module.CONFIG)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def all_inputs(module):
    x_all, _, x_rest, _ = module.SPEC.make_dataset(1.0, torch.device("cpu"), module.CONFIG)
    if len(x_rest):
        x_all = torch.cat([x_all, x_rest], dim=0)
    return x_all


@torch.no_grad()
def activation_points(model, layer_name, x):
    layer = get_module(model, layer_name)
    batches = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        batches.append(output.detach().cpu())

    handle = layer.register_forward_hook(hook)
    try:
        for start in range(0, len(x), ACTIVATION_BATCH_SIZE):
            model(x[start:start + ACTIVATION_BATCH_SIZE])
    finally:
        handle.remove()

    points = torch.cat(batches, dim=0)
    return points.reshape(points.shape[0], -1).numpy()


def weight_row_points(model, layer_name):
    modules = dict(model.named_modules())
    params = dict(model.named_parameters())

    if layer_name in modules and hasattr(modules[layer_name], "weight"):
        weight = modules[layer_name].weight.detach().cpu()
    elif layer_name in params:
        weight = params[layer_name].detach().cpu()
    else:
        print("Available module names with weight:")
        for name, layer in modules.items():
            if hasattr(layer, "weight"):
                print(f"  {name}")
        print("Available parameter names:")
        for name in params:
            print(f"  {name}")
        raise ValueError(f"Unknown layer or parameter: {layer_name}")

    if weight.ndim < 2:
        raise ValueError(f"{layer_name} is not a matrix-like weight; shape={tuple(weight.shape)}")
    return weight.reshape(weight.shape[0], -1).numpy()


def should_compute_tda_symmetry(module, model) -> bool:
    layers = tda_symmetry_layers()
    has_requested_layers = hasattr(model, "fc1") and ("fc2" not in layers or hasattr(model, "fc2"))
    return (
        INCLUDE_TDA_SYMMETRY
        and module.__name__ == "tda_grokking_mlp"
        and hasattr(module, "P")
        and has_requested_layers
    )


def tda_symmetry_layers():
    if TDA_SYMMETRY_LAYER == "both":
        return ("fc1", "fc2")
    return (TDA_SYMMETRY_LAYER,)


def tda_symmetry_encoding(model, inputs, layer):
    h = torch.relu(model.fc1(inputs))
    if layer == "fc1":
        return h
    return torch.relu(model.fc2(h))


@torch.no_grad()
def tda_first_layer_symmetry(model, p: int, layer: str) -> dict[str, float]:
    pairs = [(x, y) for x in range(p) for y in range(x + 1, p)]
    if not pairs:
        return {
            "layer": layer,
            "mean": 0.0,
            "median": 0.0,
            "q90": 0.0,
            "raw_mean": 0.0,
            "raw_median": 0.0,
            "raw_q90": 0.0,
        }

    xy = torch.zeros(len(pairs), 2 * p)
    yx = torch.zeros(len(pairs), 2 * p)
    for i, (x, y) in enumerate(pairs):
        xy[i, x] = 1.0
        xy[i, p + y] = 1.0
        yx[i, y] = 1.0
        yx[i, p + x] = 1.0

    h_xy = tda_symmetry_encoding(model, xy, layer)
    h_yx = tda_symmetry_encoding(model, yx, layer)
    diff = torch.linalg.vector_norm(h_xy - h_yx, dim=1)
    rms_norm = torch.sqrt(
        0.5 * (
            torch.linalg.vector_norm(h_xy, dim=1).square()
            + torch.linalg.vector_norm(h_yx, dim=1).square()
        )
    )
    relative_change = diff / torch.clamp(rms_norm, min=TDA_SYMMETRY_EPS)
    return {
        "layer": layer,
        "mean": float(relative_change.mean().item()),
        "median": float(relative_change.median().item()),
        "q90": float(torch.quantile(relative_change, 0.9).item()),
        "raw_mean": float(diff.mean().item()),
        "raw_median": float(diff.median().item()),
        "raw_q90": float(torch.quantile(diff, 0.9).item()),
    }


def choose_points(points):
    if MAX_POINTS is None or len(points) <= MAX_POINTS:
        return points
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(points), size=MAX_POINTS, replace=False))
    return points[idx]


def standardize(points):
    if not STANDARDIZE_POINTS:
        return points.astype(np.float64)
    points = points.astype(np.float64)
    points = points - points.mean(axis=0, keepdims=True)
    scale = points.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    return points / scale


def finite_bars(dgm):
    if len(dgm) == 0:
        return dgm
    return dgm[np.isfinite(dgm[:, 1])]


def total_persistence(dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return 0.0
    lifetimes = np.maximum(bars[:, 1] - bars[:, 0], 0.0)
    return float(np.sum(lifetimes ** TOTAL_PERSISTENCE_POWER))


def persistence_point_count(dgm):
    return len(finite_bars(dgm))


def persistence_entropy(dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return 0.0
    lifetimes = np.maximum(bars[:, 1] - bars[:, 0], 0.0)
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    probs = lifetimes / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def median_pairwise_distance(points, max_sample=1000):
    if len(points) > max_sample:
        rng = np.random.default_rng(1)
        points = points[rng.choice(len(points), size=max_sample, replace=False)]
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    dist = dist[dist > 0]
    return 0.0 if len(dist) == 0 else float(np.median(dist))


def betti_number(dgm, eps):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return 0
    return int(np.sum((bars[:, 0] <= eps) & (eps < bars[:, 1])))


def auto_betti_eps(dgm):
    bars = finite_bars(dgm)
    if len(bars) == 0:
        return None
    endpoints = np.unique(bars.reshape(-1))
    if len(endpoints) < 2:
        return float(endpoints[0])
    candidates = 0.5 * (endpoints[:-1] + endpoints[1:])
    counts = np.array([betti_number(bars, eps) for eps in candidates])
    return float(candidates[int(np.argmax(counts))])


def compute_point_cloud(module, model, x_all):
    if POINT_CLOUD == "activations":
        return activation_points(model, LAYER_NAME, x_all)
    if POINT_CLOUD == "weight_rows":
        return weight_row_points(model, LAYER_NAME)
    raise ValueError('POINT_CLOUD must be "activations" or "weight_rows"')


def compute_topology(module, run):
    x_all = all_inputs(module) if POINT_CLOUD == "activations" else None
    max_dim = max(HOMOLOGY_DIMS)
    results = []

    for i, checkpoint in enumerate(run["checkpoints"]):
        step = checkpoint["step"]
        model = load_checkpoint_model(module, checkpoint)
        points = compute_point_cloud(module, model, x_all)
        points = standardize(choose_points(points))

        print(
            f"checkpoint {i + 1}/{len(run['checkpoints'])} "
            f"step={step} points={points.shape[0]} dim={points.shape[1]}",
            flush=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The input matrix is square, but the distance_matrix flag is off.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="The input point cloud has more columns than rows; did you mean to transpose.*",
                category=UserWarning,
            )
            dgms = ripser(points, maxdim=max_dim)["dgms"]
        if BETTI_EPS_MODE == "auto":
            betti_eps = {dim: auto_betti_eps(dgms[dim]) for dim in HOMOLOGY_DIMS}
        elif BETTI_EPS_MODE == "relative":
            eps = BETTI_EPS * median_pairwise_distance(points)
            betti_eps = {dim: eps for dim in HOMOLOGY_DIMS}
        else:
            betti_eps = {dim: BETTI_EPS for dim in HOMOLOGY_DIMS}

        for dim in HOMOLOGY_DIMS:
            eps = betti_eps[dim]
            betti = 0 if eps is None else betti_number(dgms[dim], eps)
            print(f"  H{dim}: bars={len(finite_bars(dgms[dim]))} betti_eps={eps} betti={betti}", flush=True)

        tda_symmetry = None
        if should_compute_tda_symmetry(module, model):
            tda_symmetry = {}
            for layer in tda_symmetry_layers():
                layer_symmetry = tda_first_layer_symmetry(model, module.P, layer)
                tda_symmetry[layer] = layer_symmetry
                print(
                    f"  tda symmetry {layer} relative change: "
                    f"mean={layer_symmetry['mean']:.4g} "
                    f"median={layer_symmetry['median']:.4g} "
                    f"q90={layer_symmetry['q90']:.4g} "
                    f"raw_mean={layer_symmetry['raw_mean']:.4g}",
                    flush=True,
                )

        results.append({
            "step": step,
            "points_shape": points.shape,
            "dgms": dgms,
            "betti_eps": betti_eps,
            "tda_symmetry": tda_symmetry,
        })

    return results


def plot_diagrams(tda, out_dir):
    for dim in HOMOLOGY_DIMS:
        cols = min(5, len(tda))
        rows = math.ceil(len(tda) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)
        for ax, item in zip(axes.ravel(), tda):
            dgm = finite_bars(item["dgms"][dim])
            if len(dgm):
                ax.scatter(dgm[:, 0], dgm[:, 1], s=12, alpha=0.75)
                lo = float(np.min(dgm))
                hi = float(np.max(dgm))
                ax.plot([lo, hi], [lo, hi], color="black", lw=0.8, alpha=0.5)
            ax.set_title(f"step {item['step']}")
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
        for ax in axes.ravel()[len(tda):]:
            ax.axis("off")
        fig.suptitle(f"Persistence diagrams H{dim}")
        fig.tight_layout()
        path = out_dir / f"persistence_diagrams_H{dim}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_metric(tda, out_dir, metric_name, metric_fn, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    steps = [item["step"] for item in tda]
    x = plot_steps(steps)
    for dim in HOMOLOGY_DIMS:
        values = [metric_fn(item["dgms"][dim]) for item in tda]
        ax.plot(x, values, marker="o", lw=1.8, label=f"H{dim}")
    if X_AXIS_LOG10:
        ax.set_xscale("log")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{metric_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def metric_values(tda, dim, metric_fn):
    return [metric_fn(item["dgms"][dim]) for item in tda]


def betti_values(tda, dim):
    values = []
    for item in tda:
        eps = item["betti_eps"][dim] if isinstance(item["betti_eps"], dict) else item["betti_eps"]
        values.append(0 if eps is None else betti_number(item["dgms"][dim], eps))
    return values


def plot_steps(steps):
    steps = np.asarray(steps, dtype=float)
    if not X_AXIS_LOG10 or not len(steps) or np.all(steps > 0):
        return steps
    positive = steps[steps > 0]
    replacement = max(1e-6, positive.min() * 0.5) if len(positive) else 1e-6
    return np.where(steps <= 0, replacement, steps)


def aligned_checkpoint_metric(runs, key):
    steps = sorted({ckpt["step"] for run in runs for ckpt in run["checkpoints"]})
    curves = []
    for run in runs:
        values = {ckpt["step"]: ckpt[key] for ckpt in run["checkpoints"]}
        curves.append([values.get(step, np.nan) for step in steps])
    return steps, np.asarray(curves, dtype=float)


def plot_curve_summary(ax, x, curves, color, label, linestyle="-", lw=2.3):
    curves = np.asarray(curves, dtype=float)
    mean = np.nanmean(curves, axis=0)
    if CURVE_STYLE == "mean_std":
        std = np.nanstd(curves, axis=0)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
    else:
        for row in curves:
            ax.plot(x, row, color=color, alpha=0.18, lw=0.9, linestyle=linestyle)
    ax.plot(x, mean, color=color, lw=lw, linestyle=linestyle, label=label)


def plot_learning_curves(runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for condition in ("grok", "memorise"):
        cond_runs = [run for run in runs if run["condition"] == condition]
        if not cond_runs:
            continue

        for split, linestyle in (("train", "-"), ("test", "--")):
            loss_steps, losses = aligned_checkpoint_metric(cond_runs, f"{split}_loss")
            acc_steps, accs = aligned_checkpoint_metric(cond_runs, f"{split}_acc")
            loss_x = plot_steps(loss_steps)
            acc_x = plot_steps(acc_steps)
            color = CONDITION_COLORS[condition]

            plot_curve_summary(axes[0], loss_x, losses, color, f"{condition} {split}", linestyle=linestyle)
            plot_curve_summary(axes[1], acc_x, accs, color, f"{condition} {split}", linestyle=linestyle)

    if X_AXIS_LOG10:
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Checkpoint step")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Checkpoint step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    title_suffix = "mean +/- std" if CURVE_STYLE == "mean_std" else "all runs + mean"
    fig.suptitle(f"Learning curves: {title_suffix}")
    fig.tight_layout()
    path = out_dir / "loss_accuracy_all_runs.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def accuracy_curves_by_condition(runs, split):
    out = {}
    for condition in ("grok", "memorise"):
        cond_runs = [run for run in runs if run["condition"] == condition]
        if not cond_runs:
            continue
        steps, curves = aligned_checkpoint_metric(cond_runs, f"{split}_acc")
        out[condition] = (steps, curves)
    return out


def first_crossing_step(steps, values, threshold):
    for step, value in zip(steps, values):
        if value >= threshold:
            return step
    return None


def grokking_range(runs):
    grok_runs = [run for run in runs if run["condition"] == "grok"]
    if not grok_runs:
        return None

    steps, curves = aligned_checkpoint_metric(grok_runs, "test_acc")
    mean = np.nanmean(curves, axis=0)
    if len(mean) == 0:
        return None

    low = float(np.nanmin(mean))
    high = float(np.nanmax(mean))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None

    start_threshold = low + GROKKING_RANGE_START_FRAC * (high - low)
    end_threshold = low + GROKKING_RANGE_END_FRAC * (high - low)
    start = first_crossing_step(steps, mean, start_threshold)
    end = first_crossing_step(steps, mean, end_threshold)
    if start is None or end is None:
        return None
    return start, end, start_threshold, end_threshold


def mark_grokking_range(axes, runs):
    rng = grokking_range(runs)
    if rng is None:
        return

    start, end, start_threshold, end_threshold = rng
    start_x, end_x = plot_steps([start, end])
    for i, ax in enumerate(axes):
        ax.axvline(start_x, color="#16a34a", lw=1.5, linestyle=":", label="grok start" if i == 0 else None)
        ax.axvline(end_x, color="#f59e0b", lw=1.5, linestyle=":", label="grok end" if i == 0 else None)
        if i == 0:
            ax.legend(loc="best")

    print(
        f"Grokking range: step {start} to {end} "
        f"(mean test acc thresholds {start_threshold:.3f}, {end_threshold:.3f})",
        flush=True,
    )


def aligned_curves(run_tdas, dim, value_fn):
    steps = sorted({item["step"] for tda in run_tdas.values() for item in tda})
    curves = []
    labels = []
    for label, tda in run_tdas.items():
        values = {item["step"]: value for item, value in zip(tda, value_fn(tda, dim))}
        curves.append([values.get(step, np.nan) for step in steps])
        labels.append(label)
    return steps, labels, np.asarray(curves, dtype=float)


def plot_all_runs_metric(run_tdas, out_dir, metric_name, value_fn, ylabel):
    for dim in HOMOLOGY_DIMS:
        steps, labels, curves = aligned_curves(run_tdas, dim, value_fn)
        x = plot_steps(steps)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        plot_curve_summary(ax, x, curves, "#64748b", "mean", lw=2.5)
        if X_AXIS_LOG10:
            ax.set_xscale("log")
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(ylabel)
        title_suffix = "mean +/- std" if CURVE_STYLE == "mean_std" else "all runs + mean"
        ax.set_title(f"{metric_name.replace('_', ' ').title()} H{dim}: {title_suffix}")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{metric_name}_all_runs_H{dim}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def condition_from_label(label):
    return "memorise" if label.startswith("memorise") else "grok"


def plot_condition_comparison_metric(run_tdas, out_dir, metric_name, value_fn, ylabel):
    for dim in HOMOLOGY_DIMS:
        steps = sorted({item["step"] for tda in run_tdas.values() for item in tda})
        x = plot_steps(steps)
        fig, ax = plt.subplots(figsize=(8, 4.8))

        for condition in ("grok", "memorise"):
            labels = [label for label in run_tdas if condition_from_label(label) == condition]
            if not labels:
                continue

            curves = []
            for label in labels:
                tda = run_tdas[label]
                values = {item["step"]: value for item, value in zip(tda, value_fn(tda, dim))}
                row = [values.get(step, np.nan) for step in steps]
                curves.append(row)
            plot_curve_summary(ax, x, curves, CONDITION_COLORS[condition], f"{condition} mean", lw=2.6)

        if X_AXIS_LOG10:
            ax.set_xscale("log")
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric_name.replace('_', ' ').title()} H{dim}: grok vs memorise")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{metric_name}_grok_vs_memorise_H{dim}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_metric_panel(ax, run_tdas, dim, value_fn, ylabel):
    steps = sorted({item["step"] for tda in run_tdas.values() for item in tda})
    x = plot_steps(steps)

    for condition in ("grok", "memorise"):
        labels = [label for label in run_tdas if condition_from_label(label) == condition]
        if not labels:
            continue

        curves = []
        for label in labels:
            tda = run_tdas[label]
            values = {item["step"]: value for item, value in zip(tda, value_fn(tda, dim))}
            row = [values.get(step, np.nan) for step in steps]
            curves.append(row)
        plot_curve_summary(ax, x, curves, CONDITION_COLORS[condition], f"{condition} mean")

    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def has_tda_symmetry(run_tdas):
    return any(item.get("tda_symmetry") is not None for tda in run_tdas.values() for item in tda)


def tda_symmetry_values(tda, _dim):
    return [
        tda_symmetry_value(item, tda_symmetry_layers()[0])
        for item in tda
    ]


def tda_symmetry_value(item, layer, metric="mean"):
    symmetry = item.get("tda_symmetry")
    if symmetry is None:
        return np.nan
    if layer in symmetry:
        return symmetry[layer].get(metric, np.nan)
    if symmetry.get("layer") == layer:
        return symmetry.get(metric, np.nan)
    return np.nan


def plot_symmetry_panel(ax, run_tdas, metric="mean"):
    steps = sorted({item["step"] for tda in run_tdas.values() for item in tda})
    x = plot_steps(steps)
    linestyles = {"fc1": "-", "fc2": "--"}

    for layer in tda_symmetry_layers():
        for condition in ("grok", "memorise"):
            labels = [label for label in run_tdas if condition_from_label(label) == condition]
            if not labels:
                continue

            curves = []
            for label in labels:
                tda = run_tdas[label]
                values = {item["step"]: tda_symmetry_value(item, layer, metric) for item in tda}
                curves.append([values.get(step, np.nan) for step in steps])
            if not curves or np.all(np.isnan(np.asarray(curves, dtype=float))):
                continue
            plot_curve_summary(
                ax,
                x,
                curves,
                CONDITION_COLORS[condition],
                f"{condition} {layer}",
                linestyle=linestyles[layer],
            )

    ax.set_ylabel("Symmetry change")
    ax.legend(loc="best")


def plot_accuracy_panel(ax, runs):
    for split, linestyle in (("train", "-"), ("test", "--")):
        for condition, (steps, curves) in accuracy_curves_by_condition(runs, split).items():
            x = plot_steps(steps)
            plot_curve_summary(
                ax,
                x,
                curves,
                CONDITION_COLORS[condition],
                f"{condition} {split}",
                linestyle=linestyle,
            )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best")


def plot_stacked_point_count_summary(runs, run_tdas, out_dir):
    for dim in HOMOLOGY_DIMS:
        include_symmetry = has_tda_symmetry(run_tdas)
        n_panels = 5 if include_symmetry else 4
        fig_height = 13 if include_symmetry else 11
        fig, axes = plt.subplots(n_panels, 1, figsize=(9, fig_height), sharex=True)

        plot_accuracy_panel(axes[0], runs)
        plot_metric_panel(axes[1], run_tdas, dim, lambda tda, d: metric_values(tda, d, persistence_point_count), f"PD point count H{dim}")
        plot_metric_panel(axes[2], run_tdas, dim, lambda tda, d: metric_values(tda, d, total_persistence), f"Total persistence H{dim}")
        plot_metric_panel(axes[3], run_tdas, dim, lambda tda, d: metric_values(tda, d, persistence_entropy), f"Persistence entropy H{dim}")
        if include_symmetry:
            plot_symmetry_panel(axes[4], run_tdas)
        mark_grokking_range(axes, runs)

        if X_AXIS_LOG10:
            for ax in axes:
                ax.set_xscale("log")
        axes[-1].set_xlabel("Checkpoint step")
        fig.suptitle(f"Learning and topology summary H{dim}")
        fig.tight_layout()
        path = out_dir / f"stacked_summary_H{dim}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_betti(tda, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    steps = [item["step"] for item in tda]
    x = plot_steps(steps)
    for dim in HOMOLOGY_DIMS:
        values = []
        for item in tda:
            eps = item["betti_eps"][dim] if isinstance(item["betti_eps"], dict) else item["betti_eps"]
            values.append(0 if eps is None else betti_number(item["dgms"][dim], eps))
        ax.plot(x, values, marker="o", lw=1.8, label=f"H{dim}")
    if X_AXIS_LOG10:
        ax.set_xscale("log")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel("Betti number")
    ax.legend(title=f"eps mode={BETTI_EPS_MODE}, value={BETTI_EPS}")
    fig.tight_layout()
    path = out_dir / "betti_numbers.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    validate_settings()
    module = importlib.import_module(EXPERIMENT_MODULE)
    data_path = resolve_data_path(module)
    data = load_pickle(data_path)
    warn_if_data_mismatch(data, module, data_path)
    runs = selected_runs(data)
    counts = sorted({len(run["checkpoints"]) for run in runs})
    print(f"Loaded training data from {data_path}", flush=True)
    print(f"Selected {len(runs)} runs; checkpoint counts: {counts}", flush=True)

    base_dir = base_out_dir(data_path)
    base_dir.mkdir(parents=True, exist_ok=True)
    persistence_path = persistence_diagrams_path(base_dir)

    if LOAD_PERSISTENCE_DIAGRAMS and persistence_path.exists():
        print(f"Loading persistence diagrams from {persistence_path}", flush=True)
        run_tdas = load_persistence_diagrams(persistence_path)
    else:
        if LOAD_PERSISTENCE_DIAGRAMS:
            print(f"Persistence diagram cache not found at {persistence_path}; recomputing.", flush=True)

        run_tdas = {}
        for run in runs:
            print(f"\n=== {run_label(run)} ===", flush=True)
            run_tdas[run_label(run)] = compute_topology(module, run)
        save_persistence_diagrams(persistence_path, data_path, runs, run_tdas)

    plot_stacked_point_count_summary(runs, run_tdas, base_dir)


if __name__ == "__main__":
    main()
