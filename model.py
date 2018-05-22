import torch.nn as nn
from torch import FloatTensor, LongTensor
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchtext.data import Batch


class NeuralMachineTranslator(nn.Module):

    def __init__(self,
        embedding_dimension,
        vocabulary_size,
        sentence_length,
        dropout,
        input_size_decoder,
        output_size_decoder,
        hidden_size_decoder,
        batch_size,
        EOS_index,
        SOS_index,
        max_prediction_length

    ):
        super(NeuralMachineTranslator, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.sentence_length = sentence_length
        self.dropout = dropout
        self.input_size_decoder = input_size_decoder
        self.output_size_decoder = output_size_decoder
        self.hidden_size_decoder = hidden_size_decoder
        self.dropout = nn.Dropout(p=dropout)
        self.batch_size = batch_size
        self.encoder = PositionalEncoder(embedding_dimension, vocabulary_size, sentence_length, dropout)
        self.attention = Attention(embedding_dimension)
        self.decoder = Decoder(input_size_decoder, hidden_size_decoder, output_size_decoder, dropout, SOS_index)
        self.softmax = nn.LogSoftmax(dim=2)
        self.criterion = nn.NLLLoss(size_average=False, reduce=False)
        self.hidden = None
        self.context = None
        self.start = True
        self.EOS = EOS_index
        self.SOS = SOS_index
        self.max_prediction_length = max_prediction_length

    def forward(self, input: Batch, get_loss=False, teacher_forcing=False):

        input_sentences = input.src[0]
        input_lengths = input.src[1]

        batch_size = input_sentences.size()[0]

        target_sentences = input.trg[0]
        target_lengths = input.trg[1]

        french_sentence_length = input_sentences.size()[1]
        english_sentence_length = target_sentences.size()[1]

        batch_size, sentence_length = input_sentences.size()
        encoder_outputs = []
        average_embedding = Variable(FloatTensor(torch.zeros(2 * self.embedding_dimension))).repeat(batch_size, 1)
        for word in range(french_sentence_length):
            positional_embedding = self.encoder(input_sentences[:, word], word + 1)
            positional_embedding = self.dropout(positional_embedding)
            encoder_outputs.append(positional_embedding)
            average_embedding += positional_embedding / sentence_length

        if self.start:
            self.hidden = Variable(torch.unsqueeze(average_embedding, 0))
            self.context = Variable(torch.unsqueeze(average_embedding, 0))
            self.start = False

        # detach recurrent states from history for better performance during backprop
        self.hidden = repackage_hidden(self.hidden)
        self.context = repackage_hidden(self.context)

        loss = 0

        if teacher_forcing:
            predicted_sentence = np.zeros((batch_size, english_sentence_length))
        else:
            predicted_sentence = np.zeros((batch_size, self.max_prediction_length))

        has_eos = np.array([False] * batch_size)
        word = -1
        output = []
        while not all(has_eos):

            word += 1
            if teacher_forcing and word >= english_sentence_length: break
            if word >= self.max_prediction_length: break

            self.hidden = self.attention(encoder_outputs, self.hidden)

            if teacher_forcing:
                if word == 0:
                    gold_standard = Variable(LongTensor([self.SOS])).repeat(batch_size)
                else:
                    gold_standard = target_sentences[:, word - 1]
                output = torch.unsqueeze(torch.unsqueeze(gold_standard, 0), 2).float()

            output, self.hidden, self.context = self.decoder(
                output,
                self.hidden,
                self.context,
                teacher_forcing
            )

            output = self.softmax(output)

            predicted_sentence[:, word] = torch.argmax(torch.squeeze(output, 0), 1)

            has_eos |= (predicted_sentence[:, word] == self.EOS)

            if get_loss:
                mask = torch.zeros((batch_size, english_sentence_length))
                for sentence in range(batch_size):
                    sentence_mask = [0] * int(target_lengths[sentence]) \
                                    + [1] * (english_sentence_length - int(target_lengths[sentence]))
                    mask[sentence, :] = torch.LongTensor(sentence_mask)

                if not word >= english_sentence_length:
                    batch_loss = self.criterion(torch.squeeze(output), target_sentences[:, word])
                    batch_loss.masked_fill_(Variable(mask[:, word].byte()), 0.)
                    loss += batch_loss.sum() / batch_size
                else:
                    break
            else:
                loss = None

            if not teacher_forcing:
                output = torch.argmax(torch.squeeze(output, 0), 1)

        return predicted_sentence, loss


class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dimension, vocabulary_size, sentence_length, dropout):
        super(PositionalEncoder, self).__init__()

        self.sentence_length = sentence_length
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size

        self.dropout = nn.Dropout(p=dropout)

        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.relu = nn.ReLU
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

        attention_weights = F.softmax(attention_weights, dim=1)

        weighted_sum = Variable(FloatTensor(torch.zeros(embedding_dimension))).repeat(batch_size, 1)
        for i in range(sentence_length):
            weighted_sum += torch.squeeze(torch.unsqueeze(attention_weights[:, i], 1) * input[i])

        concatenation = torch.cat((torch.unsqueeze(weighted_sum, 0), hidden), 2)
        result = self.attention_layer(concatenation)

        return result


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout, SOS_index):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=200, hidden_size=hidden_size)
        self.lstm2output = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(input_size, 200)
        self.dropout = nn.Dropout(p=dropout)
        self.SOS = SOS_index

    def forward(self, input, hidden, context, teacher_forcing):

        if teacher_forcing:
            input = self.embedding(torch.squeeze(input.long()))
        else:
            batch_size = hidden.size()[1]
            if len(input) == 0:
                input = Variable(LongTensor([self.SOS])).repeat(batch_size)
                input = self.embedding(input.long())
            else:
                input = self.embedding(input.long())

        input = self.dropout(input)
        result, (hidden, context) = self.lstm(torch.unsqueeze(input, 0), (hidden, context))
        output = self.lstm2output(result)

        return output, hidden, context


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    return Variable(h.data)
