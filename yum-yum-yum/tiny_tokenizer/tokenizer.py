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

        print(self.vocab)


