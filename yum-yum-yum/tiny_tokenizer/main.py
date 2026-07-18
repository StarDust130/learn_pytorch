from tokenizer import Tokenizer

tokenizer = Tokenizer()

tokenizer.train("data.txt")

text = "I love Python"

ids = tokenizer.encode(text)
print("Encoded:", ids)

decoded = tokenizer.decode(ids)
print("Decoded:", decoded)
