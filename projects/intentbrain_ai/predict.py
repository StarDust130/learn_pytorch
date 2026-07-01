import torch

from model import IntentBrain

from data import training_data
from utils import build_vocab

vocab = build_vocab(training_data)

model = IntentBrain(
    vocab_size=len(vocab),
    embedding_dim=16,
    num_classes=4
)

model.load_state_dict(torch.load("model.pth"))
model.eval()
