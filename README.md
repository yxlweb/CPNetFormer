# CPNetFormer: Multi-Scale Temporal Graph Transformer for Computing Power Network Traffic Prediction

This repository contains the official implementation of **CPNetFormer**, a multi-scale temporal graph transformer tailored for traffic prediction in computing power networks (CPNs).

## Architecture Overview

CPNetFormer comprises five core modules:

| Module | File | Description |
|--------|------|-------------|
| **Adaptive Graph Learner** | `models.py` → `AdaptiveGraphLearner` | Infers a sparse, data-driven adjacency matrix from learnable node embeddings and temporally aggregated features. Combines static similarity, dynamic similarity, and a trainable base adjacency, then applies Top-K sparsification with temperature-scaled softmax. |
| **Multi-Scale Temporal Encoder** | `models.py` → `MultiScaleTemporalEncoder` | Parallel dilated 1-D convolutions (dilation rates 1, 2, 4) that capture traffic patterns at multiple temporal resolutions, inspired by ASPP. Features are concatenated and fused via a 1×1 convolution. |
| **Adaptive Sparse Graph Convolution** | `models.py` → `AdaptiveSparseGraphConv` | Multi-head attention where the learned adjacency serves as a structural bias and sparse mask, enabling efficient spatial message passing among nodes. |
| **Spatio-Temporal Interaction** | `models.py` → `SpatioTemporalInteraction` | Cross-modal gating mechanism: temporal features gate spatial features and vice versa, then the enhanced representations are concatenated and projected through a fusion layer with LayerNorm. |
| **Predictor Head** | `models.py` → `CPNetFormer.predictor` / `CPNetFormerLite.predictor` | A linear projection that maps the fused spatio-temporal representation to the full prediction horizon in a single forward pass (many-to-many, non-autoregressive). |

Additionally, `CPNetFormerLite` is a variant designed for univariate (single-node) time series, which treats temporal segments as virtual nodes and applies temporal graph learning.

## Repository Structure

```
.
├── main.py              # Main entry: training, evaluation, and experiment pipelines
├── models.py            # All model architectures and components
├── plot.py              # Visualization utilities (heatmaps, topology, predictions)
├── requirements.txt     # Python dependencies
├── dataset/
│   ├── ec/              # EC (European City Core Network) dataset
│   ├── uk/              # ISP (UK Academic Backbone) dataset
│   ├── Abilene/         # Abilene Internet2 backbone dataset
│   └── traffic-matrices-anonymized-v2/  # TMV2 (GÉANT) dataset
└── figures/             # Generated figures
```

## Datasets

| Dataset | Nodes | Edges | Features | Interval | Source |
|---------|-------|-------|----------|----------|--------|
| EC      | 1 (aggregated) | — | 1 | 5 min | European city core ISP |
| ISP     | 1 (aggregated) | — | 1 | 5 min | UK academic backbone |
| Abilene | 12 | 15 | 2 (in/out) | 5 min | Internet2 backbone |
| TMV2    | 23 | 37 | 2 (in/out) | 15 min | GÉANT research network |

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training & Evaluation

**Run full paper experiments** (EC + ISP, prediction lengths 48/96/128):
```bash
python main.py --experiment full_paper --batch_size 128 --lr_max 0.001 --lr_min 0.0001
```

**Single dataset experiment:**
```bash
# EC dataset, 96-step prediction
python main.py --experiment single --dataset ec --seq_len 96 --pred_len 96 --epochs 300

# ISP dataset
python main.py --experiment single --dataset isp --seq_len 96 --pred_len 96 --epochs 300

# Abilene dataset (uses log1p transform + Huber loss by default)
python main.py --experiment single --dataset abilene --seq_len 96 --pred_len 96 --epochs 100 --transform log1p --loss huber

# TMV2 dataset (includes prior adjacency from physical topology)
python main.py --experiment single --dataset tmv2 --seq_len 96 --pred_len 96 --epochs 100 --transform log1p --loss huber
```

**Interpretability analysis** (generates heatmaps and topology visualizations):
```bash
python main.py --experiment interpret --epochs 100 --loss mse --patience 10 --weight_decay 0.0001 --batch_size 128
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--experiment` | `single` | Experiment mode: `single`, `full_paper`, `pred_length`, `interpret` |
| `--dataset` | `ec` | Dataset: `ec`, `isp`, `abilene`, `tmv2`, `both` |
| `--seq_len` | `96` | Input sequence length |
| `--pred_len` | `96` | Prediction horizon |
| `--epochs` | `300` | Number of training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr_max` | `0.001` | Maximum learning rate (cosine annealing) |
| `--lr_min` | `0.0001` | Minimum learning rate |
| `--transform` | `auto` | Data transform: `none`, `log1p`, `auto` |
| `--loss` | `auto` | Loss function: `mse`, `mae`, `huber`, `auto` |
| `--patience` | `None` | Early stopping patience |

## Evaluation Metrics

- **MSE** (Mean Squared Error) — computed on standardized data
- **MAE** (Mean Absolute Error) — computed on standardized data

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.9.0
- NumPy ≥ 1.19.0
- pandas ≥ 1.3.0
- scikit-learn ≥ 0.24.0
- matplotlib ≥ 3.4.0

## Citation

If you find this work useful, please cite:


## License

This project is released for academic research purposes.
