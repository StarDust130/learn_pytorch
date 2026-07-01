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

from data import training_data
from utils import (
    build_vocab,
    sentence_to_ids,
)



vocab = build_vocab(training_data ,)
# print(vocab)

sentence = "I Want Refund"

ids = sentence_to_ids(
    sentence,
    vocab
)

print(ids)
