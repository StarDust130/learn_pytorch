import torch

from model import PositionalEmbedding

layer = PositionalEmbedding()

x = torch.randn(2, 10, 64)

output = layer(x)

print(output.shape)
