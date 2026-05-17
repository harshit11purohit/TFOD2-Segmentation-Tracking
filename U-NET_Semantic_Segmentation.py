import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset


class SyntheticShapes(Dataset):
    def __init__(self, n=48, size=96, seed=7):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.items = [self._make_item(i) for i in range(n)]

    def _make_item(self, i):
        rng = np.random.default_rng(1000 + i)
        image = Image.new("RGB", (self.size, self.size), (25, 35, 45))
        mask = Image.new("L", (self.size, self.size), 0)
        draw_img = ImageDraw.Draw(image)
        draw_mask = ImageDraw.Draw(mask)
        x0 = int(rng.integers(12, 42))
        y0 = int(rng.integers(12, 42))
        x1 = int(rng.integers(55, 86))
        y1 = int(rng.integers(55, 86))
        color = tuple(int(v) for v in rng.integers(110, 245, size=3))
        if i % 2 == 0:
            draw_img.ellipse((x0, y0, x1, y1), fill=color)
            draw_mask.ellipse((x0, y0, x1, y1), fill=1)
        else:
            draw_img.rectangle((x0, y0, x1, y1), fill=color)
            draw_mask.rectangle((x0, y0, x1, y1), fill=1)
        image_arr = np.asarray(image, dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.float32)
        return image_arr, mask_arr

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, mask = self.items[idx]
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self.block(3, 16)
        self.enc2 = self.block(16, 32)
        self.middle = self.block(32, 64)
        self.dec2 = self.block(96, 32)
        self.dec1 = self.block(48, 16)
        self.out = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def block(inp, out):
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out, out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        m = self.middle(F.max_pool2d(e2, 2))
        d2 = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def binary_iou(logits, target):
    pred = torch.sigmoid(logits) > 0.5
    target = target > 0.5
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return (intersection / union).item() if union.item() else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Train on generated shape masks.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    if not args.demo:
        raise SystemExit("Use --demo for this verified tiny U-Net exercise. Adapt it to Oxford Pets after download.")

    torch.manual_seed(3)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SyntheticShapes()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = TinyUNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        losses = []
        for image, mask in loader:
            logits = model(image)
            loss = loss_fn(logits, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"epoch={epoch + 1} loss={np.mean(losses):.4f}")

    image, mask = dataset[0]
    with torch.no_grad():
        logits = model(image.unsqueeze(0))
        pred = torch.sigmoid(logits)[0, 0].numpy()
        iou = binary_iou(logits, mask.unsqueeze(0))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(image.permute(1, 2, 0).numpy())
    axes[0].set_title("Input")
    axes[1].imshow(mask[0].numpy(), cmap="gray")
    axes[1].set_title("Target")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction IoU={iou:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "02_tiny_unet_prediction.png", dpi=160)
    print(f"sample_iou={iou:.4f}")
    print(f"Saved: {out_dir / '02_tiny_unet_prediction.png'}")


if __name__ == "__main__":
    main()

