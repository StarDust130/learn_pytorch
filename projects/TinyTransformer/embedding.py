import torch
import torch.nn as nn

embedding = nn.Embedding(
    num_embeddings=30,
    embedding_dim=4
)

token_ids = torch.tensor([2, 5, 7])

vectors = embedding(token_ids)

print(vectors)


position_embedding = nn.Embedding(
    num_embeddings=128,
    embedding_dim=32
)

positions = torch.arange(5)

position_vectors = position_embedding(positions)
