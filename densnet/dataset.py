"""
Dataset + augmentation for bladder segmentation.

With only 20-100 images, augmentation is not optional — it's what makes
training viable at all. Each epoch the same image is shown under many
random transforms (flips, rotations, brightness/contrast shifts,
elastic warps), so the network effectively sees far more variety than
the raw count suggests and is much less likely to just memorise the
handful of training images.
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


IMG_SIZE = 256  # network input size; images are resized to this square


def _list_pairs(images_dir, masks_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    pairs = []
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith(exts):
            continue
        stem = os.path.splitext(fn)[0]
        mask_path = os.path.join(masks_dir, stem + "_mask.png")
        if os.path.exists(mask_path):
            pairs.append((os.path.join(images_dir, fn), mask_path))
    return pairs


def train_augmentations():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(p=0.3),
        A.ElasticTransform(alpha=40, sigma=6, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def val_augmentations():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class BladderDataset(Dataset):
    def __init__(self, images_dir, masks_dir, pairs=None, train=True):
        self.pairs = pairs if pairs is not None else _list_pairs(images_dir, masks_dir)
        self.tf = train_augmentations() if train else val_augmentations()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        out = self.tf(image=img, mask=mask)
        img_t = torch.from_numpy(out["image"].transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(out["mask"]).unsqueeze(0).float()
        return img_t, mask_t


def make_splits(images_dir, masks_dir, val_frac=0.2, seed=42):
    pairs = _list_pairs(images_dir, masks_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            f"No image/mask pairs found. Did you run step1_generate_masks.py? "
            f"(looked in {images_dir} + {masks_dir})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    return train_pairs, val_pairs
