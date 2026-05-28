# Gearbox Graph Ablation 2026

Experiment comparing two graph construction strategies for vibration-based fault detection with Graph Attention Networks (GAT) on the EGB-250 gearbox dataset.

## Research question

Does the choice of graph topology (k-NN in statistical feature space vs. Pearson correlation between sensor channels) significantly affect GAT fault classification performance?

## Graph construction strategies

| Strategy | Nodes | Edges | Motivation |
|---|---|---|---|
| k-NN (baseline) | time windows | Euclidean k-NN (k=8) in 54-d statistical feature space | Windows with similar statistical profiles share the same operating condition |
| Pearson Top-k | sensor channels | Top-4 absolute Pearson correlations per channel (symmetrized) | Mechanical faults propagate vibration through structural paths, creating physically interpretable inter-sensor couplings |

## Statistical design

- **20 independent runs** per strategy (seeds 42–61)
- **95% bootstrap confidence intervals** (2000 resamples, percentile method)
- **Mann-Whitney U test** (two-sided, α=0.05) — non-parametric, no normality assumption

## Repository structure

```
src/
  models/gat.py          — VibrationGAT architecture
  preprocessing.py       — build_knn_graph + build_pearson_graph
  data_loader.py         — EGB-250 .mat loader
notebooks/
  pearson_graph_experiment.ipynb  — main experiment (Colab-ready)
results/comparison/              — generated after running the notebook
  graph_ablation.csv             — per-run metrics
  graph_ablation_summary.csv     — mean, std, CI95 per strategy
  graph_ablation_meta.json       — hyperparameters + statistical test results
```

> `data/`, `models/`, and `results/` are excluded from version control (see `.gitignore`).  
> The preprocessed EGB-250 arrays (`data/processed/`) must be provided separately.

## Running on Google Colab

1. Open `notebooks/pearson_graph_experiment.ipynb` in Colab.
2. Cell 1 installs PyTorch Geometric if needed.
3. Cell 3 mounts Google Drive. Update `DRIVE_PROCESSED_PATH` to point to your `data/processed/` folder before running.
4. Run all cells — approximately 35 min on T4/V100.

## Requirements

```
numpy pandas scipy scikit-learn torch torch-geometric h5py matplotlib seaborn networkx ipython
```

Install with:

```bash
pip install -r requirements.txt
```

> `torch-geometric` requires a build compatible with your installed PyTorch and CUDA version.  
> See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Authors

Pedro Camara
