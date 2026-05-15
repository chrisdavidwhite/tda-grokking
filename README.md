# Topological Data Analysis of Grokking in Neural Networks

This notebook applies **persistent homology** to study the *grokking* phenomenon — the delayed generalisation that can occur when training small neural networks on algorithmic tasks.

## What it does

Small MLPs (2 hidden layers, 128 units each) are trained on modular arithmetic (addition mod 47) under two conditions:

- **Grokking** (weight decay = 5.0): networks initially memorise, then suddenly generalise to ~97% test accuracy
- **Memorisation** (weight decay = 0): networks overfit and never generalise

The weight matrices from each layer are treated as point clouds, and **Vietoris–Rips persistent homology** (via `ripser`) is computed at each training checkpoint. This extracts topological features — connected components (H₀) and loops (H₁) — that characterise the geometry of the learned representations.

## Key findings

- Grokked networks develop significantly richer loop structure (H₁) in their weight topology compared to memorised networks
- Wasserstein distances between final persistence diagrams cleanly separate the two solution types; hierarchical clustering recovers the ground-truth grouping
- Topological signatures emerge gradually during training and are most pronounced in the second weight layer (W₂)
- PCA of flattened weight vectors shows grokked and memorised training trajectories diverge in parameter space

## Outputs

The notebook saves 11 figures covering learning curves, TDA dynamics during training, persistence diagrams, barcodes, Wasserstein distance heatmaps, MDS embeddings, layer-wise comparisons, and correlation with test accuracy.

## Dependencies

```
torch ripser persim scipy scikit-learn matplotlib numpy
```
