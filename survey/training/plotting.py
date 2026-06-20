from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/codex_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/codex_xdg_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def condition_runs(data, condition):
    return [row for row in data["results"] if row["condition"] == condition]


def plot_mean_std(ax, x, values, color, label):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    ax.plot(x, mean, color=color, lw=2.2, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def plot_mean_all(ax, x, values, color, label):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    for row in values:
        ax.plot(x, row, color=color, alpha=0.22, lw=0.9)
    ax.plot(x, mean, color=color, lw=2.4, label=label)


def aligned_checkpoint_values(runs, key):
    steps = sorted({ckpt["step"] for run in runs for ckpt in run["checkpoints"]})
    by_step = []
    for run in runs:
        values = {ckpt["step"]: ckpt[key] for ckpt in run["checkpoints"]}
        by_step.append([values.get(step, np.nan) for step in steps])
    return steps, by_step


def checkpoint_weights(ckpt):
    if "weights" in ckpt:
        return ckpt["weights"]
    if "state_dict" not in ckpt:
        raise KeyError('Checkpoint has neither "weights" nor "state_dict".')
    return np.concatenate([
        value.detach().cpu().numpy().reshape(-1)
        for value in ckpt["state_dict"].values()
    ]).astype(np.float32, copy=False)


def plot_loss_accuracy(data, out_dir, curve_style="mean_std"):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"grok": "#1d4ed8", "memorise": "#dc2626"}
    if curve_style not in {"mean_std", "mean_all"}:
        raise ValueError('curve_style must be "mean_std" or "mean_all"')
    plot_curve = plot_mean_all if curve_style == "mean_all" else plot_mean_std

    for condition in ("grok", "memorise"):
        runs = condition_runs(data, condition)
        if not runs:
            continue
        for split, linestyle in (("train", "-"), ("test", "--")):
            loss_steps, losses = aligned_checkpoint_values(runs, f"{split}_loss")
            acc_steps, accs = aligned_checkpoint_values(runs, f"{split}_acc")
            label = f"{condition} {split}"
            plot_curve(axes[0], loss_steps, losses, colors[condition], label)
            axes[0].lines[-1].set_linestyle(linestyle)
            plot_curve(axes[1], acc_steps, accs, colors[condition], label)
            axes[1].lines[-1].set_linestyle(linestyle)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend()

    title_suffix = "mean + all runs" if curve_style == "mean_all" else "mean +/- std across runs"
    fig.suptitle(f"Train/test outcomes: {title_suffix}")
    fig.tight_layout()
    path = out_dir / "loss_accuracy_mean_std.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def plot_weight_cdfs(data, out_dir):
    cdf_dir = out_dir / "weight_cdfs"
    cdf_dir.mkdir(parents=True, exist_ok=True)

    for run in data["results"]:
        fig, ax = plt.subplots(figsize=(7, 5.5))
        checkpoints = run["checkpoints"]
        cmap = plt.get_cmap("viridis")
        for i, ckpt in enumerate(checkpoints):
            w = np.sort(checkpoint_weights(ckpt))
            y = np.linspace(0.0, 1.0, len(w), endpoint=True)
            color = cmap(i / max(1, len(checkpoints) - 1))
            label = f"{ckpt['step']}" if i in (0, len(checkpoints) - 1) else None
            ax.plot(w, y, color=color, alpha=0.8, lw=1.0, label=label)
        ax.set_title(f"Weight empirical CDF: {run['condition']} run {run['run']}")
        ax.set_xlabel("Weight value")
        ax.set_ylabel("Empirical CDF")
        ax.legend(title="step", loc="lower right")
        fig.tight_layout()
        path = cdf_dir / f"cdf_{run['condition']}_run{run['run']}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved CDF plot to {path}")


def cdf_values(weights, x_grid):
    return np.searchsorted(np.sort(weights), x_grid, side="right") / len(weights)


def plot_mean_weight_cdfs(data, out_dir, x_log_scale=True):
    runs_by_condition = {condition: condition_runs(data, condition) for condition in ("grok", "memorise")}
    all_weights = np.concatenate([
        checkpoint_weights(ckpt)
        for runs in runs_by_condition.values()
        for run in runs
        for ckpt in run["checkpoints"]
    ])

    if x_log_scale:
        max_abs = max(float(np.max(np.abs(all_weights))), 1e-12)
        x_grid = np.geomspace(1e-8, max_abs, 600)
        x_label = "Absolute weight value"
    else:
        x_grid = np.linspace(float(np.min(all_weights)), float(np.max(all_weights)), 600)
        x_label = "Weight value"

    fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True, sharey=True)
    colors = {"grok": "#1d4ed8", "memorise": "#dc2626"}
    n_ckpts = min(len(run["checkpoints"]) for runs in runs_by_condition.values() for run in runs)

    for ax, condition in zip(axes, ("grok", "memorise")):
        for ckpt_idx in range(n_ckpts):
            alpha = 0.18 + 0.72 * ckpt_idx / max(1, n_ckpts - 1)
            runs = runs_by_condition[condition]
            if x_log_scale:
                cdfs = [cdf_values(np.abs(checkpoint_weights(run["checkpoints"][ckpt_idx])), x_grid) for run in runs]
            else:
                cdfs = [cdf_values(checkpoint_weights(run["checkpoints"][ckpt_idx]), x_grid) for run in runs]
            ax.plot(x_grid, np.asarray(cdfs).mean(axis=0), color=colors[condition], alpha=alpha, lw=1.0)
        ax.set_ylabel("Empirical CDF")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{condition}: mean CDF across runs")
        ax.plot([], [], color=colors[condition], alpha=0.25, label="earlier checkpoints")
        ax.plot([], [], color=colors[condition], alpha=0.9, label="later checkpoints")
        ax.legend()

    if x_log_scale:
        axes[-1].set_xscale("log")
    axes[-1].set_xlabel(x_label)
    fig.suptitle("Mean weight CDF at every checkpoint")
    fig.tight_layout()
    path = out_dir / "mean_weight_cdfs_all_checkpoints.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean CDF plot to {path}")


def plot_all(data, out_dir, cdf_x_log_scale=True, curve_style="mean_std"):
    plot_loss_accuracy(data, out_dir, curve_style=curve_style)
    plot_weight_cdfs(data, out_dir)
    plot_mean_weight_cdfs(data, out_dir, x_log_scale=cdf_x_log_scale)
