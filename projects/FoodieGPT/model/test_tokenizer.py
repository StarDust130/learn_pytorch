from tokenizer import CharacterTokenizer

text = "Hello FoodieGPT"

tokenizer = CharacterTokenizer(text)

tokens = tokenizer.encode(text)

print(tokens)

print(tokenizer.decode(tokens))