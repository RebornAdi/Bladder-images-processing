import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from models.unet import UNet
from dataset_seg import SegDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SegDataset("data/processed/images", "data/masks")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    total_loss = 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)

        preds = model(imgs)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")