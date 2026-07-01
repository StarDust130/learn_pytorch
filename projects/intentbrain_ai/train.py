"""
intentbrain-ai/

│
├── train.py          ⭐ Main file (Run this)
├── model.py          🧠 Neural Network
├── data.py           📚 Dataset
├── utils.py          🔧 Helper functions
├── predict.py        🤖 Test AI later
├── requirements.txt
└── .venv

"""

import time
import torch
import torch.nn as nn

from data import training_data
from utils import (
    build_vocab,
    sentence_to_ids,
    label_map
)
from model import IntentBrain


# -----------------------
# Vocabulary
# -----------------------
vocab = build_vocab(training_data)

print("Vocabulary Size:", len(vocab))


# -----------------------
# Model
# -----------------------
model = IntentBrain(
    vocab_size=len(vocab),
    embedding_dim=16,
    num_classes=4
)


# -----------------------
# Teacher
# -----------------------
loss_fn = nn.CrossEntropyLoss()


# -----------------------
# Mechanic
# -----------------------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01
)


# -----------------------
# Training
# -----------------------

best_loss = float("inf")
start_time = time.time()

for epoch in range(1000):

    total_loss = 0

    for sentence, label in training_data:

        x = sentence_to_ids(sentence, vocab)
        y = torch.tensor([label_map[label]])

        prediction = model(x)
        prediction = prediction.unsqueeze(0)

        loss = loss_fn(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if total_loss < best_loss:
        best_loss = total_loss
        icon = "⭐"
    else:
        icon = " "

    if epoch % 100 == 0:

        elapsed = time.time() - start_time
        progress = (epoch + 1) / 10000

        bar = "█" * int(progress * 30)
        bar += "-" * (30 - len(bar))

        print("\n" + "=" * 60)
        print("🧠 IntentBrain AI Training")
        print("=" * 60)

        print(f"Epoch      : {epoch:,} / 10,00")
        print(f"Progress   : [{bar}] {progress*100:.1f}%")
        print(f"Loss       : {total_loss:.6f}")
        print(f"Best Loss  : {best_loss:.6f} {icon}")
        print(f"Runtime    : {elapsed:.1f}s")

        print("=" * 60)

print("\nTraining Finished ✅")
torch.save(model.state_dict(), "model.pth")


# -----------------------
# Test
# -----------------------

while True:

    sentence = input("\nYou: ")

    x = sentence_to_ids(sentence, vocab)

    with torch.no_grad():

        output = model(x)

        prediction = int(torch.argmax(output).item())

    labels = list(label_map.keys())

    print("Intent:", labels[prediction])
