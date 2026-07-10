from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

from data import training_data

label_map = {
    "Greeting": 0,
    "Food": 1,
    "Travel": 2,
    "Refund": 3,
    "Music": 4,
    "Alarm": 5,
    "Calculator": 6,
    "Weather": 7,
    "News": 8,
}


class IntentDataset(Dataset):

    def __init__(self):
        self.data = training_data

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    def __len__(self): #Total length
        return len(self.data)

    def __getitem__(self, index):

        text, label = self.data[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=16,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_map[label]),
        }
