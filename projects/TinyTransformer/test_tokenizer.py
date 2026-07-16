from tokenizer import CharacterTokenizer

with open("dataset.txt", "r") as f:
    text = f.read()

tokenizer = CharacterTokenizer(text)

print("Vocabulary")

print(tokenizer.stoi)

print()

encoded = tokenizer.encode("Pizza")

print(encoded)

print()

print(tokenizer.decode(encoded))
