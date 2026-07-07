'''
Dataset's job is: Store -> Retrieve -> Preprocess -> Return


'''

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

from data import training_data

# Label -> Number
label_map = {
    "Funny": 0,
    "Normal": 1,
    "Dead": 2
}


class MemeDataset(Dataset):

    def __init__(self):

        self.data = training_data

        self.tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        text, label = self.data[index]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=10,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label_map[label])
        }
