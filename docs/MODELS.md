# DeepGuard AI — Model roadmap

## Phase 1 — Baseline
**EfficientNet-B0 / B2**

Fast, strong ImageNet transfer. Primary model for the academic baseline.

## Phase 2 — Stronger CNN
**XceptionNet** (implemented via ResNeXt-50 until `timm` Xception is installed)

Higher capacity CNN for comparison.

## Phase 3 — Research
**Vision Transformer (ViT-B/16)**

Transformer baseline for the performance comparison table.

## Expected comparison table (fill from real experiments)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| EfficientNet | — | — | — | — | — |
| Xception | — | — | — | — | — |
| ViT | — | — | — | — | — |

Use `scripts/evaluate.py` after training. Do not invent metrics.
