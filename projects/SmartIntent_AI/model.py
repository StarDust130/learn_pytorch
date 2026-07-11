import math
import torch
import torch.nn as nn

# ✍️ Built from scratch
# ✅ Ready option: nn.Embedding
class PositionalEmbedding(nn.Module):
    # 📍 Adds word position

    def __init__(self, max_length=128, embedding_dim=64):
        super().__init__()

        # 📦 Store position vectors
        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

    def forward(self, x):
        # 📏 Get sentence length
        sequence_length = x.size(1)

        # 🔢 Make position numbers
        positions = torch.arange(sequence_length)

        # 📱 Move to same device
        positions = positions.to(x.device)

        # 📍 Get position vectors
        position_vectors = self.position_embedding(
            positions
        )

        # ➕ Add position to words
        return x + position_vectors

# ❌❌❌WE are not using it❌❌❌, we use MultiHeadAttention
# ✍️ Built from scratch
# 🚀 Better: nn.MultiheadAttention
class SelfAttention(nn.Module):

    def __init__(self, embedding_dim=64):

        super().__init__()

        # ❓ Query
        self.query = nn.Linear(embedding_dim, embedding_dim)

        # 🔑 Key
        self.key = nn.Linear(embedding_dim, embedding_dim)

        # 💎 Value
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x):

        # ✨ Make Q K V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 🤝 Compare words
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        # ⚖️ Scale scores
        scores = scores / math.sqrt(Q.size(-1))

        # 🎯 Get attention
        attention = torch.softmax(
            scores,
            dim=-1
        )

        # 🧠 Mix information
        output = torch.matmul(
            attention,
            V
        )

        return output

# ✅ Production version 🏆🏆( This use in production)
# 🚀 PyTorch built-in
class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim=64,
        num_heads=4
    ):

        super().__init__()

        # 👀 Many attention heads
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):

        # 🧠 Learn from many views
        output, _ = self.attention(
            x,
            x,
            x
        )

        return output
 
# ✍️ Built from scratch
class FeedForward(nn.Module):

    def __init__(self, embedding_dim=64):

        super().__init__()

        # 🔄 Small neural network
        self.network = nn.Sequential(

            # 📈 Make bigger
            nn.Linear(
                embedding_dim,
                embedding_dim * 4
            ),

            # ⚡ Add power
            nn.GELU(),

            # 📉 Make smaller
            nn.Linear(
                embedding_dim * 4,
                embedding_dim
            )

        )

    def forward(self, x):

        return self.network(x)

# ✍️ Built from scratch
# 🚀 Better: nn.TransformerEncoderLayer
class TransformerBlock(nn.Module):

    def __init__(self, embedding_dim=64):

        super().__init__()

        # 👀 Attention
        self.attention = MultiHeadAttention(
            embedding_dim
        )

        # 📏 Normalize
        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        # 🧠 Think more
        self.feed_forward = FeedForward(
            embedding_dim
        )

        # 📏 Normalize again
        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        # 👀 Look at all words
        attention_output = self.attention(x)

        # ➕ Add + normalize
        x = self.norm1(
            x + attention_output
        )

        # 🧠 Learn more
        ff_output = self.feed_forward(x)

        # ➕ Add + normalize
        x = self.norm2(
            x + ff_output
        )

        return x



# ✍️ Built from scratch
# 🚀 Better: BERT, RoBERTa, DistilBERT
# OG MODEL 🤭
class SmartIntentAI(nn.Module):

    def __init__(
        self,
        vocab_size=30522,
        embedding_dim=64,
        max_length=128,
        num_classes=9
    ):

        super().__init__()

        # 📚 Word vectors
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # 📍 Position vectors
        self.position = PositionalEmbedding(
            max_length,
            embedding_dim
        )

        # 🧠 Main brain
        self.transformer = TransformerBlock(
            embedding_dim
        )

        # 🎯 Final answer
        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

    def forward(self, input_ids):

        # 📚 Words → vectors
        x = self.embedding(input_ids)

        # 📍 Add positions
        x = self.position(x)

        # 🧠 Understand sentence
        x = self.transformer(x)

        # ⭐ First token
        x = x[:, 0]

        # 🎯 Predict class
        output = self.classifier(x)

        return output


"""
😊 Easy Flow

Sentence 📝
   ↓
Tokenizer ✂️
   ↓
Token IDs 🔢
   ↓
Embedding 📚
(Word meaning)
   ↓
Position 📍
(Word place)
   ↓
Transformer 🧠
(Learn sentence)
   ↓
Classifier 🎯
(Pick class)
   ↓
Prediction ✅


💡 Easy Words

📚 Embedding = Turns words into numbers.

📍 Position = Tells where each word is.

❓ Query = What this word wants.

🔑 Key = What this word has.

💎 Value = Information to share.

👀 Attention = Finds important words.

🧠 Transformer = Understands the sentence.

🎯 Classifier = Chooses the final category.
"""
