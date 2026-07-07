from dataset import MemeDataset
from torch.utils.data import DataLoader

from model import MemeBrain

dataset = MemeDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

model = MemeBrain()

batch = next(iter(loader))

output = model(batch["input_ids"])

print(output)
print(output.shape)
