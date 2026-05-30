# Gearbox Graph Ablation 2026

Experiment comparing two graph construction strategies for vibration-based fault detection with Graph Attention Networks (GAT) on the EGB-250 gearbox dataset.

## Research question

Holding everything else constant (nodes, node features, k, architecture, seeds, split), does the **edge-construction criterion** alone — Euclidean k-NN vs. Pearson correlation between sensor channels — significantly affect GAT fault classification performance?

## Graph construction strategies

Both strategies build a per-window graph over the **same 9 sensor channels** (nodes), each carrying the **same 6 per-channel statistics** (RMS, kurtosis, crest factor, peak-to-peak, skewness, zero-crossing rate), with the **same k=4**. The *only* difference is how edges are drawn. This isolates the edge metric as the single experimental variable; a Jaccard edge-overlap check (~0.49) confirms the two graph sets are genuinely distinct.

| Strategy | Nodes | Edges | Relation captured |
|---|---|---|---|
| k-NN (Euclidean) | 9 sensor channels | each channel → its 4 nearest channels in Euclidean distance over the statistical feature space (symmetrized) | channels with similar statistical profiles |
| Pearson Top-k | 9 sensor channels | each channel → its 4 channels of highest absolute Pearson correlation within the window (symmetrized) | channels that co-vary in time |

## Statistical design

- **20 independent runs** per strategy (seeds 42–61)
- **95% bootstrap confidence intervals** (2000 resamples, percentile method)
- **Mann-Whitney U test** (two-sided, α=0.05) — non-parametric, no normality assumption

## Result

With the edge metric isolated as the only variable, the two constructions are **statistically equivalent**: no significant difference in F1 (U=251, p=0.172) or AUC (U=246, p=0.218).

| Strategy | F1 macro (95% CI) | AUC (95% CI) |
|---|---|---|
| k-NN (Euclidean) | 0.9289 [0.9071, 0.9525] | 0.9902 [0.9865, 0.9938] |
| Pearson Top-k | 0.9091 [0.8822, 0.9358] | 0.9852 [0.9794, 0.9909] |

The choice of channel-connection criterion is not decisive for GAT fault diagnosis on EGB-250; both relations capture comparable discriminative information.

## Repository structure

```
src/
  models/gat.py          — VibrationGAT architecture
  preprocessing.py       — build_knn_channel_graph + build_pearson_graph
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
