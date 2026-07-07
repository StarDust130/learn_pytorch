import torch.nn as nn


class MemeBrain(nn.Module):

    def __init__(self):

        super().__init__()

        # Convert IDs → Meaning Vectors
        self.embedding = nn.Embedding(
            num_embeddings=30522,
            embedding_dim=64
        )

        # Brain
        self.fc1 = nn.Linear(64, 32)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(32, 3)

    def forward(self, input_ids):

        # IDs → Vectors
        x = self.embedding(input_ids)

        # Many Word Vectors → One Sentence Vector
        x = x.mean(dim=1)

        # Think
        x = self.fc1(x)

        x = self.relu(x)

        # Predict
        x = self.fc2(x)

        return x
