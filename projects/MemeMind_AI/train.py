from dataset import MemeDataset
from torch.utils.data import DataLoader

dataset = MemeDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True
)

for batch in loader:

    print(batch)

    break
