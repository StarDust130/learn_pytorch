import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    #  model supports sentences up to 128 tokens

    def __init__(self, max_length=128, embedding_dim=64):
        super().__init__()
      
    # Position -> Embedding 🤯
    # It create 128 row each has 64 numbers
        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

    def forward(self, x):
        sequence_length = x.size(1)
        positions = torch.arange(sequence_length)
        positions = positions.to(x.device)
        position_vectors = self.position_embedding(
            positions
        )
        return x + position_vectors


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim=64):

        super().__init__()

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        return Q, K, V
