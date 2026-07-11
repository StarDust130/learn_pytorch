import torch
from transformers import AutoTokenizer

from model import SmartIntentAI

# ===================================================
# 🧠 Labels
# ===================================================

labels = [
    "👋 Greeting",
    "🍕 Food",
    "✈️ Travel",
    "💸 Refund",
    "🎵 Music",
    "⏰ Alarm",
    "🧮 Calculator",
    "☀️ Weather",
    "📰 News"
]

# ===================================================
# 🤖 Load Tokenizer
# ===================================================

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

# ===================================================
# 🧠 Load Model
# ===================================================

model = SmartIntentAI()

model.load_state_dict(
    torch.load(
        "smart_intent_ai.pth",
        map_location="cpu"
    )
)

model.eval()

# ===================================================
# 🚀 SmartIntent AI
# ===================================================

print("=" * 60)
print("🤖 SmartIntent AI")
print("=" * 60)
print("💬 Type any sentence.")
print("🚪 Type 'exit' to quit.")
print("=" * 60)

# ===================================================
# 💬 Chat
# ===================================================

while True:

    text = input("\n🧑 You : ").strip()

    # 🚪 Exit
    if text.lower() in ["exit", "quit", "bye"]:
        print("\n👋 Goodbye! See you again.")
        break

    # ❌ Empty input
    if text == "":
        print("⚠️ Please type something.")
        continue

    encoding = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=16
    )

    with torch.no_grad():

        prediction = model(
            encoding["input_ids"]
        )

        confidence = torch.softmax(
            prediction,
            dim=1
        )

        # ensure intent is a Python int for safe list indexing
        intent = int(confidence.argmax(dim=1).item())

        score = confidence.max().item() * 100

    print(f"🤖 Intent     : {labels[intent]}")
    print(f"📊 Confidence : {score:.2f}%")
