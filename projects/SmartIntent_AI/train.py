from model import SmartIntentAI
from dataset import IntentDataset

from torch.utils.data import DataLoader

import torch
import torch.nn as nn

# 📦 Dataset
dataset = IntentDataset()

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

# 🧠 Model
model = SmartIntentAI()

# 👨‍🏫 Teacher
loss_fn = nn.CrossEntropyLoss()

# 🔧 Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4
)

# ===================================================
# ===================================================
# Traning Loop 🚂🚃🚃🚃
epochs = 20

for epoch in range(epochs):

    total_loss = 0

    for batch in loader:

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

    print(
        f"Epoch {epoch+1} | Loss = {total_loss:.4f}"
    )

# ==========================================================
# ==========================================================
# 📊 Evaluation
correct = 0
total = 0

model.eval()

with torch.no_grad():

    for batch in loader:

        prediction = model(
            batch["input_ids"]
        )

        predicted = prediction.argmax(dim=1)

        correct += (
            predicted == batch["label"]
        ).sum().item()

        total += len(predicted)

accuracy = correct / total

print(f"Accuracy : {accuracy:.2%}")



# ========================================================
# ========================================================
# 💾 Save
torch.save(
    model.state_dict(),
    "smart_intent_ai.pth"
)

print("✅ Model Saved")
