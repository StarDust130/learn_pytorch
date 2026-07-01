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

from utils import tokenize
from data import training_data
from utils import build_vocab

# print(training_data)

vocab = build_vocab(training_data)
print(vocab)

sentence = "I Want Refund"

tokens = tokenize(sentence)

# print(tokens)
