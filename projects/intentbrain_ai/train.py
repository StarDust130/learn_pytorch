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
for epoch in range(500):

    total_loss = 0

    for sentence, label in training_data:

        # Convert sentence -> tensor
        x = sentence_to_ids(sentence, vocab)

        # Convert label -> tensor
        y = torch.tensor([label_map[label]])

        # AI Guess
        prediction = model(x)

        # CrossEntropy expects batch dimension
        prediction = prediction.unsqueeze(0)

        # Teacher
        loss = loss_fn(prediction, y)

        # Remove old gradients
        optimizer.zero_grad()

        # Find mistakes
        loss.backward()

        # Improve model
        optimizer.step()

        total_loss += loss.item()

    if epoch % 50 == 0:

        print(
            f"Epoch {epoch} | Loss = {total_loss:.4f}"
        )


print("\nTraining Finished ✅")


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
