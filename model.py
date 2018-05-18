import torch.nn as nn
from torch import FloatTensor, LongTensor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, dimension, embedding_dimension, vocabulary_size, sentence_length, dropout, log=False):
        super(PositionalEncoder, self).__init__()

        self.sentence_length = sentence_length
        self.dimension = dimension
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.dropout = nn.Dropout

        self.log = log
        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.relu = nn.ReLU
        # self.positional_encodings = Variable(FloatTensor(self.precompute_positional_encoding()))
        self.max_positions = 100
        self.positional_embedding = nn.Embedding(self.max_positions, embedding_dimension)

    def forward(self, input, input_position):

        batch_size = input.size()[0]
        embedding = self.input_embedding(input)

        positions = Variable(LongTensor(np.array([input_position]))).repeat(batch_size, 1)
        positional_encoding = self.positional_embedding(positions)
        positional_encoding = torch.squeeze(positional_encoding, 1)
        positional_embedding = torch.cat((embedding, positional_encoding), 1)

        return positional_embedding

    # def precompute_positional_encoding(self):
    #
    #     if self.log:
    #         print("Precomputing positional encodings..")
    #
    #     positional_encodings = np.zeros((self.sentence_length, self.embedding_dimension))
    #     for pos in range(self.sentence_length, 2):
    #         embedding_vector = np.array(range(1, self.embedding_dimension + 1)) / self.embedding_dimension
    #         positional_encoding_vector = np.ones((self.embedding_dimension, 1)) * pos / 10000**embedding_vector
    #         even_positional_encoding_vector = np.sin(positional_encoding_vector)
    #         uneven_positional_encoding_vector = np.cos(positional_encoding_vector)
    #         positional_encodings[pos] = even_positional_encoding_vector
    #         positional_encodings[pos + 1] = uneven_positional_encoding_vector
    #
    #     return positional_encodings


class Attention(nn.Module):

    def __init__(self, embedding_dimension):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(4 * embedding_dimension, 2 * embedding_dimension)


    def forward(self, input, hidden):

        sentence_length = len(input)
        batch_size, embedding_dimension = input[0].size()

        # attention
        attention_weights = Variable(FloatTensor(torch.zeros(sentence_length))).repeat(batch_size, 1)

        for i in range(sentence_length):
            attention_weight = torch.bmm(input[i].view(batch_size, 1, embedding_dimension), hidden.view(batch_size, embedding_dimension, 1))
            attention_weights[:, i] = torch.squeeze(torch.squeeze(attention_weight, 1), 1)

        attention_weights = F.softmax(attention_weights)

        weighted_sum = Variable(FloatTensor(torch.zeros(embedding_dimension))).repeat(batch_size, 1)
        for i in range(sentence_length):
            weighted_sum += torch.squeeze(torch.unsqueeze(attention_weights[:, i], 1) * input[i])

        concatenation = torch.cat((torch.unsqueeze(weighted_sum, 0), hidden), 2)
        result = self.attention_layer(concatenation)

        return result


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.lstm2output = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, context):

        result, (hidden, context) = self.lstm(input, (hidden, context))
        output = self.lstm2output(result)

        return output, hidden, context


