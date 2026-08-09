"""Training script scaffold for DeepGuard AI models.

Expected dataset layout (FaceForensics++ / Celeb-DF style):

    data/
      train/
        real/
        fake/
      val/
        real/
        fake/
      test/
        real/
        fake/

Usage:
    python scripts/train.py --model efficientnet --data-dir ./data --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures.factory import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepGuard deepfake classifier")
    p.add_argument("--model", default="efficientnet", choices=["efficientnet", "xception", "vit"])
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=ROOT / "backend" / "weights")
    return p.parse_args()


def make_loaders(data_dir: Path, image_size: int, batch_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    # Ensure class order: fake=1 preferred. ImageFolder sorts alphabetically: fake, real
    print("Classes:", train_ds.class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    train_loader, val_loader = make_loaders(args.data_dir, args.image_size, args.batch_size)

    model = build_model(args.model, pretrained=True).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.model}.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * y.size(0)
        train_loss = running / len(train_loader.dataset)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc >= best:
            best = val_acc
            torch.save({"state_dict": model.state_dict(), "model": args.model}, out_path)
            print(f"  saved best checkpoint → {out_path}")

    print(f"Done. Best val accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
