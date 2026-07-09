import torch

from transformers import AutoTokenizer

from model import MemeBrain

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

model = MemeBrain()

model.load_state_dict(

    torch.load("memebrain.pth")
)

model.eval()

text = "I am cool boy"

encoding = tokenizer(

    text,

    truncation=True,

    padding="max_length",

    max_length=10,

    return_tensors="pt"

)

with torch.no_grad():

    output = model(

        encoding["input_ids"]

    )

prediction = output.argmax(dim=1).item()

labels = {

    0: "Funny",

    1: "Normal",

    2: "Dead"

}

print(

    labels[prediction]

)
