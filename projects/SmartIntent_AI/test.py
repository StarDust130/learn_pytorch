"""
➡️ Total = 9 Intent Categories 
👋 Greeting 
🍕 Food 
✈️ Travel 
💸 Refund 
🎵 Music 
⏰ Alarm 
🧮 Calculator 
☀️ Weather 
📰 News

"""

# import torch

# from model import PositionalEmbedding

# layer = PositionalEmbedding()

# x = torch.randn(2, 10, 64)

# output = layer(x)

# print(output.shape)

import torch

from model import SelfAttention

layer = SelfAttention()

x = torch.randn(2, 10, 64)

Q, K, V = layer(x)

print(Q.shape)
print(K.shape)
print(V.shape)
