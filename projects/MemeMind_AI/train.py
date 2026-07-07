import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MemeDataset
from model import MemeBrain


# ======================
# Data
# ======================

dataset = MemeDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

# ======================
# Model
# ======================

model = MemeBrain()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

EPOCHS = 20

print("=" * 55)
print(" MemeBrain Training")
print("=" * 55)
print(f"Dataset Size : {len(dataset)}")
print(f"Batches      : {len(loader)}")
print(f"Epochs       : {EPOCHS}")
print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")
print("=" * 55)


for epoch in range(EPOCHS):

    start = time.time()

    model.train()

    total_loss = 0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch+1:02}/{EPOCHS}",
        leave=False
    )

    for batch in progress:

        input_ids = batch["input_ids"]
        labels = batch["label"]

        prediction = model(input_ids)

        loss = loss_fn(
            prediction,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        avg_loss = total_loss / (progress.n + 1)

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            avg=f"{avg_loss:.4f}"
        )

    epoch_time = time.time() - start

    print(
        f"[{epoch+1:02}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time:.2f}s"
    )

print("\nTraining Complete.")
