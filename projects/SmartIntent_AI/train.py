from model import SmartIntentAI
from dataset import IntentDataset, label_map
from torch.utils.data import DataLoader, random_split


import torch
import torch.nn as nn

# ===================================================
# ⚙️ Settings
# ===================================================

EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ===================================================
# 📦 Dataset
# ===================================================

# ===================================================
# 📦 Dataset
# ===================================================

dataset = IntentDataset()

# ✂️ Split Dataset (70 / 15 / 15)
train_size = int(0.70 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# 🚚 DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ===================================================
# 🧠 Model
# ===================================================

model = SmartIntentAI().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

# ===================================================
# 📊 Project Info
# ===================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("=" * 60)
print("🚀 SmartIntent AI")
print("=" * 60)

print(f"📦 Dataset Size        : {len(dataset)}")
print(f"🎯 Intent Classes      : {len(label_map)}")
print(f"📚 Batch Size          : {BATCH_SIZE}")
print(f"🔄 Epochs              : {EPOCHS}")
print(f"📈 Learning Rate       : {LEARNING_RATE}")
print(f"⚡ Device              : {device.type.upper()}")
print(f"🧠 Optimizer           : {optimizer.__class__.__name__}")
print(f"👨‍🏫 Loss Function      : {loss_fn.__class__.__name__}")
print(f"📊 Total Parameters    : {total_params:,}")
print(f"✅ Trainable Params    : {trainable_params:,}")

print("=" * 60)

# ===================================================
# 🚂 Training
# ===================================================

# 💾 Save Best Model
best_val_accuracy = 0

print("\n🚂 Training Started...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        prediction = model(input_ids)

        loss = loss_fn(
            prediction,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predicted = prediction.argmax(dim=1)

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    train_accuracy = correct / total * 100

    avg_loss = total_loss / len(train_loader)

    # ===================================================
    # 🔍 Validation
   # ===================================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            prediction = model(input_ids)

            predicted = prediction.argmax(dim=1)

            val_correct += (
                predicted == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total * 100

    print(
        f"📘 Epoch {epoch+1:02}/{EPOCHS}"
        f" | Loss: {avg_loss:.4f}"
        f" | Train: {train_accuracy:.2f}%"
        f" | Val: {val_accuracy:.2f}%"
    )

    # 💾 Save Best Model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "smart_intent_ai.pth"
        )

        print("💾 Best Model Saved!")



# ===================================================
# 📊 Evaluation
# ===================================================

print("\n" + "=" * 60)
print("🧪 Testing Best Model...")
print("=" * 60)

# 📥 Load Best Model
model.load_state_dict(
    torch.load(
        "smart_intent_ai.pth",
        map_location=device
    )
)

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        prediction = model(input_ids)

        predicted = prediction.argmax(dim=1)

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

accuracy = correct / total * 100

print(f"🏆 Test Accuracy : {accuracy:.2f}%")
print(f"🥇 Best Validation Accuracy : {best_val_accuracy:.2f}%")

print("=" * 60)

# ===================================================
# 💾 Save Model
# ===================================================

torch.save(
    model.state_dict(),
    "smart_intent_ai.pth"
)

print("💾 Model Saved Successfully!")
print("📁 smart_intent_ai.pth")

print("=" * 60)
print("🎉 Training Complete!")
print("=" * 60)
