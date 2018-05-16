import torch.nn as nn
from torch import FloatTensor
import numpy as np


class PositionalEncoder(nn.Module):
    def __init__(self, dimension, embedding_dimension, vocabulary_size, sentence_length, dropout, log=False):
        super(PositionalEncoder, self).__init__()

        self.sentence_length = sentence_length
        self.dimension = dimension
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.dropout = nn.Dropout

        self.log = log
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.relu = nn.ReLU
        self.positional_encodings = FloatTensor(self.precompute_positional_encoding())

    def forward(self, input, input_position):

        embedding = self.embedding(input)
        positional_encoding = self.positional_encodings[input_position]
        positional_embedding = embedding + positional_encoding

        return positional_embedding

    def precompute_positional_encoding(self):

        if self.log:
            print("Precomputing positional encodings..")

        positional_encodings = np.zeros((self.sentence_length, self.embedding_dimension))
        for pos in range(self.sentence_length, 2):
            embedding_vector = np.array(range(1, self.embedding_dimension + 1)) / self.embedding_dimension
            positional_encoding_vector = np.ones((self.embedding_dimension, 1)) * pos / 10000**embedding_vector
            even_positional_encoding_vector = np.sin(positional_encoding_vector)
            uneven_positional_encoding_vector = np.cos(positional_encoding_vector)
            positional_encodings[pos] = even_positional_encoding_vector
            positional_encodings[pos + 1] = uneven_positional_encoding_vector

        return positional_encodings




