import torch

# Tokenize a sentence into individual words 🗼
import string

# Tokenize a sentence into individual words 🗼
def tokenize(sentence):

    sentence = sentence.lower() # Convert to lowercase

    sentence = sentence.translate(
        str.maketrans("", "", string.punctuation) # Remove punctuation
    )

    words = sentence.split() # Split the sentence into words

    return words


# Build a vocabulary from the training data 🏗️ 
# eg:- {"hello": 0, "world": 1, "I": 2, "want": 3, "refund": 4}
def build_vocab(training_data):

    vocab = {}

    index = 0

    for sentence, intent in training_data:

        words = tokenize(sentence)

        for word in words:

            if word not in vocab:

                vocab[word] = index

                index += 1

    return vocab


# Sentence to IDs 🆔
def sentence_to_ids(sentence, vocab):

    words = tokenize(sentence)

    ids = []

    for word in words:

        ids.append(vocab[word])

    return torch.tensor(ids)
