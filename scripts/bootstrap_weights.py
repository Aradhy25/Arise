"""Bootstrap a real PyTorch deepfake classifier from synthetic forgery data.

Creates face-like real/fake pairs with blending seams, double compression,
and frequency artifacts, then fine-tunes EfficientNet-B0 and writes
backend/weights/efficientnet.pth.

This is NOT mock inference — the resulting checkpoint is a trained neural net.
For research-grade accuracy, retrain later on FaceForensics++ / Celeb-DF.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.architectures.factory import build_model  # noqa: E402


def _face_canvas(rng: np.random.Generator, size: int = 224) -> np.ndarray:
    """Procedural face-like RGB image (skin tones + facial structure cues)."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    # Skin base
    skin = np.array([rng.integers(150, 210), rng.integers(110, 170), rng.integers(90, 140)], dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = size // 2, size // 2 + 10
    face = ((xx - cx) / (size * 0.32)) ** 2 + ((yy - cy) / (size * 0.40)) ** 2 <= 1.0
    img[face] = skin

    # Eyes
    for ex in (cx - size // 7, cx + size // 7):
        ey = cy - size // 10
        eye = ((xx - ex) / (size * 0.06)) ** 2 + ((yy - ey) / (size * 0.035)) ** 2 <= 1.0
        img[eye] = [40, 40, 40]
        pupil = ((xx - ex) / (size * 0.025)) ** 2 + ((yy - ey) / (size * 0.025)) ** 2 <= 1.0
        img[pupil] = [10, 10, 10]

    # Mouth
    my = cy + size // 7
    mouth = ((xx - cx) / (size * 0.12)) ** 2 + ((yy - my) / (size * 0.04)) ** 2 <= 1.0
    img[mouth] = [120, 60, 70]

    # Soft shading + natural noise
    shade = (yy / size).astype(np.float32)[..., None] * 18
    img = np.clip(img - shade, 0, 255)
    noise = rng.normal(0, 4.5, img.shape)
    img = np.clip(img + noise, 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _make_fake(real_a: np.ndarray, real_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate face-swap / blend forgery artifacts."""
    h, w = real_a.shape[:2]
    # Soft elliptical blend mask (classic deepfake tell)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2 + int(rng.integers(-8, 9)), h // 2 + int(rng.integers(-8, 9))
    mask = np.exp(-(((xx - cx) / (w * 0.28)) ** 2 + ((yy - cy) / (h * 0.34)) ** 2))
    mask = mask.astype(np.float32)[..., None]
    blended = (mask * real_b.astype(np.float32) + (1 - mask) * real_a.astype(np.float32)).astype(np.uint8)

    # Color mismatch on blend region
    shift = rng.integers(-18, 19, size=3)
    blended = np.clip(blended.astype(np.int16) + (mask * shift).astype(np.int16), 0, 255).astype(np.uint8)

    # Frequency / ghosting artifact
    warped = cv2.resize(blended, (w - 4, h - 4))
    warped = cv2.resize(warped, (w, h))
    ghost = cv2.addWeighted(blended, 0.82, warped, 0.18, 0)

    # Double JPEG compression (common in social-media deepfakes)
    ghost = _jpeg(ghost, quality=int(rng.integers(55, 80)))
    ghost = _jpeg(ghost, quality=int(rng.integers(70, 92)))

    # Local blur on seam
    seam = (mask[..., 0] > 0.35) & (mask[..., 0] < 0.75)
    blurred = cv2.GaussianBlur(ghost, (7, 7), 0)
    ghost[seam] = blurred[seam]
    return ghost


def _make_real(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light realistic distortions that should still look 'real'."""
    out = img.copy()
    if rng.random() < 0.5:
        out = _jpeg(out, quality=int(rng.integers(85, 98)))
    if rng.random() < 0.3:
        out = cv2.convertScaleAbs(out, alpha=float(rng.uniform(0.95, 1.05)), beta=int(rng.integers(-5, 6)))
    return out


class SyntheticDeepfakeDataset(Dataset):
    def __init__(self, n_samples: int, seed: int = 42, train: bool = True):
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
        self.train = train
        self.tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # Pre-generate to keep training fast/deterministic
        self.images: list[np.ndarray] = []
        self.labels: list[int] = []
        for i in range(n_samples):
            local = np.random.default_rng(seed + i * 17)
            a = _face_canvas(local)
            if i % 2 == 0:
                self.images.append(_make_real(a, local))
                self.labels.append(0)  # REAL
            else:
                b = _face_canvas(local)
                self.images.append(_make_fake(a, b, local))
                self.labels.append(1)  # FAKE

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        return self.tf(self.images[idx]), self.labels[idx]


@torch.no_grad()
def accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-samples", type=int, default=600)
    p.add_argument("--val-samples", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=Path, default=ROOT / "backend" / "weights" / "efficientnet.pth")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bootstrapping EfficientNet on {device} …")

    train_ds = SyntheticDeepfakeDataset(args.train_samples, seed=42, train=True)
    val_ds = SyntheticDeepfakeDataset(args.val_samples, seed=99, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model("efficientnet", pretrained=True).to(device)
    # Freeze early backbone for speed; train head + last blocks
    for name, param in model.backbone.named_parameters():
        if "features.6" in name or "features.7" in name or "features.8" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = torch.as_tensor(y, device=device)
            optim.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optim.step()
            running += loss.item() * y.size(0)
            n += y.size(0)
        val_acc = accuracy(model, val_loader, device)
        print(f"epoch {epoch}/{args.epochs}  loss={running/max(n,1):.4f}  val_acc={val_acc:.3f}")
        if val_acc >= best:
            best = val_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": "efficientnet",
                    "val_acc": best,
                    "source": "synthetic_bootstrap",
                },
                args.out,
            )

    print(f"Saved checkpoint → {args.out}  (best val_acc={best:.3f})")


if __name__ == "__main__":
    main()
