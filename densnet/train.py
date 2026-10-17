"""
STEP 2 — Train the DenseNet-121 U-Net on the pseudo-labelled data.

Loss = Dice + BCE (a standard, robust combo for segmentation). We track
Dice and IoU on the validation split each epoch and keep the best
checkpoint.

Usage (on your GPU machine):
    python train.py --images data/images --masks data/masks --epochs 60

Outputs:
    best_model.pth   — best-validation-Dice checkpoint
    training_log.csv — per-epoch metrics for your report's plots
"""
import os
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BladderDataset, make_splits
from model import build_model


def dice_coef(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean().item()


def iou_coef(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) >= 1).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


class DiceBCELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, target, eps=1e-6):
        bce = self.bce(logits, target)
        pred = torch.sigmoid(logits)
        inter = (pred * target).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = 1 - ((2 * inter + eps) / (union + eps)).mean()
        return bce + dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--masks", default="data/masks")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--out", default="best_model.pth")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: no GPU detected — training will be slow. This script "
              "is meant to run on your GPU machine.")

    train_pairs, val_pairs = make_splits(args.images, args.masks, args.val_frac)
    print(f"Train images: {len(train_pairs)}  Val images: {len(val_pairs)}")
    if len(train_pairs) < 8:
        print("NOTE: very few training images. Rely heavily on augmentation and "
              "watch for overfitting (val Dice plateauing while train keeps rising).")

    train_ds = BladderDataset(None, None, pairs=train_pairs, train=True)
    val_ds = BladderDataset(None, None, pairs=val_pairs, train=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = DiceBCELoss()

    best_dice = -1.0
    log_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        for img, mask in train_dl:
            img, mask = img.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(img)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * img.size(0)
        tr_loss /= len(train_ds)
        sched.step()

        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for img, mask in val_dl:
                img, mask = img.to(device), mask.to(device)
                logits = model(img)
                dices.append(dice_coef(logits, mask))
                ious.append(iou_coef(logits, mask))
        vdice, viou = float(np.mean(dices)), float(np.mean(ious))
        print(f"epoch {epoch:3d}  train_loss {tr_loss:.4f}  val_dice {vdice:.4f}  val_iou {viou:.4f}")
        log_rows.append((epoch, tr_loss, vdice, viou))

        if vdice > best_dice:
            best_dice = vdice
            torch.save(model.state_dict(), args.out)
            print(f"           ^ new best (dice {best_dice:.4f}) saved to {args.out}")

    with open("training_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_dice", "val_iou"])
        w.writerows(log_rows)
    print(f"\nBest val Dice: {best_dice:.4f}. Log written to training_log.csv")


if __name__ == "__main__":
    main()
