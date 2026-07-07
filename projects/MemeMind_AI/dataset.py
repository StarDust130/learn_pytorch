'''
Dataset's job is: Store -> Return. Store -> Return.

'''

from torch.utils.data import Dataset
from data import training_data


class MemeDataset(Dataset):  # Custom dataset

    def __init__(self):  # Setup dataset
        self.data = training_data  # Save data

    def __len__(self):  # Total items
        return len(self.data)

    def __getitem__(self, index):  # Get one item
        return self.data[index]
