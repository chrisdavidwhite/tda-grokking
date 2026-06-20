# Grokking Experiments

This repo trains grokking-style models from `networks/` and then runs offline
topological analysis on the saved checkpoints.

## Run One Experiment

Pick one file from `networks/`, then run it from the repo root:

```bash
python networks/tda_grokking_mlp.py
```

Each network file defines its own `CONFIG`, including the number of runs,
training steps, checkpoint count, and output folder.

The experiment saves a pickle under:

```text
runs/<experiment_name>/
```

For example:

```text
runs/tda_grokking_mlp/001_43_add_12321_100k_100k_260620-122341.pkl
```

## Run Topological Analysis

Open `topology_analysis.py` and edit the settings near the top:

```python
EXPERIMENT_MODULE = "tda_grokking_mlp"
RUN_SERIAL = 1
DATA_PATH = None

POINT_CLOUD = "weight_rows"  # or "activations"
LAYER_NAME = "fc2"
```

Then run:

```bash
python topology_analysis.py
```

Figures and cached persistence diagrams are written under:

```text
figures/
```

## Notes

- `RUN_SERIAL = 1` loads the saved file beginning with `001_`.
- Set `RUN_SERIAL = None` to use the newest saved run for that experiment.
- Set `DATA_PATH` to an exact `.pkl` path if you want to bypass serial lookup.
- If analysis is slow, set `MAX_RUNS_PER_CONDITION` or `MAX_POINTS` in
  `topology_analysis.py`.
