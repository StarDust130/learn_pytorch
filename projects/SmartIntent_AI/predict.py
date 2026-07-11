import torch

from transformers import AutoTokenizer
from model import SmartIntentAI

labels = [
    "Greeting",
    "Food",
    "Travel",
    "Refund",
    "Music",
    "Alarm",
    "Calculator",
    "Weather",
    "News"
]

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = SmartIntentAI()

model.load_state_dict(
    torch.load("smart_intent_ai.pth")
)

model.eval()


while True:

    text = input("You : ")

    if text.lower() == "exit":
        break

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    with torch.no_grad():

        prediction = model(
            encoding["input_ids"]
        )

        intent = prediction.argmax(dim=1).item()

    print(
        "🤖",
        labels[intent]
    )
