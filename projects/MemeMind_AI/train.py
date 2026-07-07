from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

text = "My code works first try! 😂"

encoded = tokenizer(text)

print(tokenizer.tokenize(text))

print()

print(encoded["input_ids"])
