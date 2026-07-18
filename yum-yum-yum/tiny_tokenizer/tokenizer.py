class Tokenizer:
    def __init__(self):
        self.vocab = {}

    def train(self, file_path):
        with open(file_path, "r") as file:
            for line in file:
                words = line.strip().split()

                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)

        print("Vocabulary:")
        print(self.vocab)

    def encode(self, text):
        words = text.split()
        ids = []

        for word in words:
            ids.append(self.vocab[word])

        return ids

    def decode(self, ids):
        reverse_vocab = {}

        for word, token_id in self.vocab.items():
            reverse_vocab[token_id] = word

        words = []

        for token_id in ids:
            words.append(reverse_vocab[token_id])

        return " ".join(words)
