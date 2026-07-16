
class CharacterTokenizer:

    def __init__(self, text):

        # Find every unique character and sort them.
        chars = sorted(set(text))

        # Character → Number
        self.stoi = {
            ch: i
            for i, ch in enumerate(chars)
        }

        # Number → Character
        self.itos = {
            i: ch
            for ch, i in self.stoi.items()
        }

    def encode(self, text):

        return [
            self.stoi[ch]
            for ch in text
        ]

    def decode(self, ids):

        return "".join(
            self.itos[i]
            for i in ids
        )
