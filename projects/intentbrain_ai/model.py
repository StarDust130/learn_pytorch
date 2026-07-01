import torch.nn as nn


class IntentBrain(nn.Module):

    def __init__(self, vocab_size, embedding_dim, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.linear = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, x):

        # Word IDs -> Word Vectors
        x = self.embedding(x)

        # Average sentence vector
        x = x.mean(dim=0)

        # Predict Intent
        x = self.linear(x)

        return x
