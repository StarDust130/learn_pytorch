from torch.utils.data import DataLoader

from dataset import IntentDataset

dataset = IntentDataset()

loader = DataLoader(

    dataset,

    batch_size=32,

    shuffle=True

)

batch = next(iter(loader))

print(batch["input_ids"].shape)

print(batch["attention_mask"].shape)

print(batch["label"])
