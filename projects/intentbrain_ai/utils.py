import string
import torch


# -----------------------
# Tokenizer
# -----------------------
def tokenize(sentence):

    sentence = sentence.lower()

    sentence = sentence.translate(
        str.maketrans("", "", string.punctuation)
    )

    return sentence.split()


# -----------------------
# Vocabulary
# -----------------------
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


# -----------------------
# Sentence -> Tensor
# -----------------------
def sentence_to_ids(sentence, vocab):

    words = tokenize(sentence)

    ids = []

    for word in words:

        ids.append(vocab[word])

    return torch.tensor(ids)


# -----------------------
# Intent Labels
# -----------------------
label_map = {

    "Track": 0,
    "Refund": 1,
    "Cancel": 2,
    "Password": 3

}
