import torch
import torch.nn as nn

class PostionalEncoding(nn.Module):
    #  model supports sentences up to 128 tokens

    def __int__(self, max_length=128, embedding_dim=64):
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


