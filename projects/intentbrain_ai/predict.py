import torch

from model import IntentBrain

from data import training_data
from utils import build_vocab



from utils import (
    sentence_to_ids,
    label_map
)

vocab = build_vocab(training_data)

model = IntentBrain(
    vocab_size=len(vocab),
    embedding_dim=16,
    num_classes=4,
    
)

model.load_state_dict(torch.load("model.pth"))
model.eval()

while True:

    sentence = input("You: ")

    x = sentence_to_ids(sentence, vocab)

    with torch.no_grad():
        output = model(x)

        prediction = int(torch.argmax(output).item())

    labels = list(label_map.keys())

    print("Intent:", labels[prediction])
