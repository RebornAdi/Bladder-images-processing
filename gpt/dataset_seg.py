import cv2
import torch
from torch.utils.data import Dataset
import os

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.images = sorted(os.listdir(img_dir))
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_name), 0)
        img = cv2.resize(img, (512, 512)) / 255.0

        mask = cv2.imread(os.path.join(self.mask_dir, img_name), 0)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        img = torch.tensor(img).unsqueeze(0).float()
        mask = torch.tensor(mask).long()

        return img, mask