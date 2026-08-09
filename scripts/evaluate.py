"""Evaluate trained DeepGuard models on a held-out test set.

Produces accuracy / precision / recall / F1 / ROC-AUC and confusion matrix.
Does NOT invent metrics — only reports what the model actually scores.

Usage:
    python scripts/evaluate.py --model efficientnet --data-dir ./data --weights backend/weights/efficientnet.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures.factory import build_model, load_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="efficientnet")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts" / "eval")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = datasets.ImageFolder(args.data_dir / "test", transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = build_model(args.model, pretrained=False)
    model, loaded = load_checkpoint(model, args.weights, device)
    if not loaded:
        raise SystemExit(f"Could not load weights: {args.weights}")

    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(y.numpy().tolist())
        y_pred.extend(preds.tolist())
        y_prob.extend(probs.tolist())

    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    y_prob_a = np.array(y_prob)

    metrics = {
        "model": args.model,
        "accuracy": float(accuracy_score(y_true_a, y_pred_a)),
        "precision": float(precision_score(y_true_a, y_pred_a, zero_division=0)),
        "recall": float(recall_score(y_true_a, y_pred_a, zero_division=0)),
        "f1": float(f1_score(y_true_a, y_pred_a, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_a, y_prob_a)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true_a, y_pred_a).tolist(),
        "classes": ds.class_to_idx,
        "n_samples": len(ds),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(classification_report(y_true_a, y_pred_a, target_names=list(ds.class_to_idx.keys())))
    print(json.dumps(metrics, indent=2))

    # Confusion matrix plot
    cm = confusion_matrix(y_true_a, y_pred_a)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title(f"{args.model} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    labels = list(ds.class_to_idx.keys())
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(args.out / f"{args.model}_confusion_matrix.png", dpi=150)

    if metrics["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_true_a, y_prob_a)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"{args.model} ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / f"{args.model}_roc.png", dpi=150)

    print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()
