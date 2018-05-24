from abc import abstractmethod
from typing import List

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
        PAD_index,
        max_prediction_length
    ):
        super(NeuralMachineTranslator, self).__init__()

        # hyper parameter settings
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.sentence_length = sentence_length
        self.dropout = dropout
        self.input_size_decoder = input_size_decoder
        self.output_size_decoder = output_size_decoder
        self.hidden_size_decoder = hidden_size_decoder
        self.dropout = nn.Dropout(p=dropout)
        self.batch_size = batch_size
        self.max_prediction_length = max_prediction_length

        # indices of special tokens
        self.EOS = EOS_index
        self.SOS = SOS_index
        self.PAD = PAD_index

        # get model attributes
        # self.encoder = PositionalEncoder(embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index)
        self.encoder = PositionalEncoder(embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index)
        self.attention = Attention(embedding_dimension)
        self.decoder = Decoder(input_size_decoder, hidden_size_decoder, output_size_decoder, dropout, SOS_index, PAD_index)
        self.softmax = nn.LogSoftmax(dim=2)

        # loss function
        self.criterion = nn.CrossEntropyLoss(size_average=False, reduce=False)

        # initialize hidden states
        self.hidden = None
        self.context = None
        self.start = True

    def forward(self, input: Batch, optimizer=None, get_loss=False, teacher_forcing=False):

        # unpack batch
        input_sentences = input.src[0]
        target_sentences = input.trg[0]
        target_lengths = input.trg[1]
        french_sentence_length = input_sentences.size()[1]
        english_sentence_length = target_sentences.size()[1]
        batch_size, sentence_length = input_sentences.size()

        # encoder
        encoder_output, word_encodings = self.encoder.encode(input_sentences, sentence_length)

        # initialize hidden state and conetxt state with average embedding
        if self.start:
            self.hidden = Variable(torch.unsqueeze(encoder_output, 0), requires_grad=True)
            self.context = Variable(torch.unsqueeze(encoder_output, 0), requires_grad=True)
            self.start = False

        # detach recurrent states from history for better performance during backprop
        self.hidden = repackage_hidden(self.hidden)
        self.context = repackage_hidden(self.context)
        self.hidden = self.hidden.detach()
        self.context = self.context.detach()

        # initialize loss
        loss = 0

        # if teacher forcing is used, predicted sentence length will be maximally english sentence length
        if teacher_forcing:
            predicted_sentence_length = english_sentence_length
        else:
            predicted_sentence_length = self.max_prediction_length

        predicted_sentence = np.zeros((batch_size, predicted_sentence_length))

        # array to keep track of which sentences in the batch has reached the <EOS> token
        has_eos = np.array([False] * batch_size)
        word = -1

        # initialize output for decoder
        output = []

        # loop until all sentences in batch have reached <EOS> token
        while not all(has_eos):

            # print(word)
            word += 1

            # stop loop if prediction has certain length
            if teacher_forcing and word >= english_sentence_length: break
            if word >= self.max_prediction_length: break

            # attention
            self.hidden = self.attention(word_encodings, self.hidden)

            # if teacher forcing is used get previous gold standard word of target sentence
            if teacher_forcing:
                if word == 0:

                    # <SOS> token is initial gold standard word
                    gold_standard = Variable(LongTensor([self.SOS])).repeat(batch_size)
                else:
                    gold_standard = target_sentences[:, word - 1]
                output = torch.unsqueeze(torch.unsqueeze(gold_standard, 0), 2).float()

            # decoder
            output, self.hidden, self.context = self.decoder(
                output,
                self.hidden,
                self.context,
                teacher_forcing
            )

            # get predicted words from decoder output
            predicted_sentence[:, word] = torch.argmax(torch.squeeze(output, 0), 1)

            # updating which sentences from batch have reached <EOS> token
            has_eos |= (predicted_sentence[:, word] == self.EOS)

            if get_loss:

                # prepare mask for padding
                mask = torch.zeros((batch_size, english_sentence_length))
                for sentence in range(batch_size):
                    sentence_mask = [0] * int(target_lengths[sentence]) \
                                    + [1] * (english_sentence_length - int(target_lengths[sentence]))
                    mask[sentence, :] = torch.LongTensor(sentence_mask)

                # get loss if predicted sentence is not longer than target sentence
                if not word >= english_sentence_length:
                    batch_loss = self.criterion(torch.squeeze(output), target_sentences[:, word])
                    batch_loss.masked_fill_(Variable(mask[:, word].byte()), 0.)
                    loss += batch_loss.sum() / batch_size

                # otherwise break out of while loop since further training will not do anything
                else:
                    break
            else:
                loss = None

            # get indices for next decoder run
            if not teacher_forcing:
                output = torch.argmax(torch.squeeze(output, 0), 1)

        return predicted_sentence, loss


class Encoder(nn.Module):

    def __init__(self, embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index):
        super(Encoder, self).__init__()

        # hyper parameter settings
        self.sentence_length = sentence_length
        self.embedding_dimension = embedding_dimension
        self.vocabulary_size = vocabulary_size
        self.dropout = nn.Dropout(p=dropout)
        self.max_positions = 100
        self.pad_index = PAD_index

    @abstractmethod
    def forward(self, input: LongTensor, input_position: int):
        raise NotImplementedError

    @abstractmethod
    def encode(self, input_sentences, sentence_lengths) -> (FloatTensor, List[FloatTensor]):
        raise NotImplementedError


class GRUEncoder(Encoder):

    def __init__(self, embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index):
        super(GRUEncoder, self).__init__(embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index)

        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=PAD_index)
        self.GRU = nn.GRU(
            input_size=embedding_dimension,
            hidden_size=embedding_dimension,
            num_layers=1,
            bias=True,
            dropout=dropout,
            bidirectional=True
        )

    def forward(self, input: LongTensor, input_position: int) -> FloatTensor:

        embedding = self.input_embedding(input.t())
        output, final_hidden_states = self.GRU(embedding)

        return output, final_hidden_states

    def encode(self, input_sentences, sentence_lengths) -> (FloatTensor, List[FloatTensor]):

        sequence_length = input_sentences.size()[1]

        output, hidden_states = self.forward(input_sentences, 0)

        word_encodings = []
        for word_index in range(sequence_length):
            word_encodings.append(
                torch.squeeze(output[word_index, :, :], dim=0)
            )

        # concatenate final hidden states from left-to-right and right-to-left pass through GRU
        final_hidden_state = torch.cat([hidden_states[0, :, :], hidden_states[1, :, :]], 1)

        return final_hidden_state, word_encodings


class PositionalEncoder(Encoder):

    def __init__(self, embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index):

        super(PositionalEncoder, self).__init__(embedding_dimension, vocabulary_size, sentence_length, dropout, PAD_index)

        # layers
        self.input_embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=PAD_index)
        self.positional_embedding = nn.Embedding(self.max_positions, embedding_dimension, padding_idx=PAD_index)

    def encode(self, input_sentences, sentence_lengths) -> (FloatTensor, List[FloatTensor]):

        batch_size = input_sentences.size()[0]
        french_sentence_length = input_sentences.size()[1]

        word_encodings = []
        average_encoding = Variable(FloatTensor(torch.zeros(2 * self.embedding_dimension))).repeat(batch_size, 1)
        for word in range(french_sentence_length):
            positional_embedding = self.forward(input_sentences[:, word], word + 1)
            # positional_embedding = self.dropout(positional_embedding) #TODO: handle dropout
            word_encodings.append(positional_embedding)
            average_encoding += positional_embedding / sentence_lengths
        #TODO: cannot sum up over all words in batch and divide by actual sentence length (add 0 if token is <PAD>)

        return average_encoding, word_encodings

    def forward(self, input: LongTensor, input_position: int) -> FloatTensor:

        batch_size = input.size()[0]

        # word embedding
        embedding = self.input_embedding(input)

        # positional embedding
        positions = Variable(LongTensor(np.array([input_position]))).repeat(batch_size, 1)
        positional_encoding = self.positional_embedding(positions)
        positional_encoding = torch.squeeze(positional_encoding, 1)

        # concatenate word and positional embedding
        positional_embedding = torch.cat((embedding, positional_encoding), 1)

        return positional_embedding


class Attention(nn.Module):

    def __init__(self, embedding_dimension):
        super(Attention, self).__init__()
        self.attention_layer = nn.Linear(4 * embedding_dimension, 2 * embedding_dimension)

    def forward(self, input, hidden):

        sentence_length = len(input)

        # bsz = 1
        if len(input[0].size()) == 1:
            input[0] = torch.unsqueeze(input[0], 0)

        batch_size, embedding_dimension = input[0].size()

        # get attention weights
        attention_weights = Variable(FloatTensor(torch.zeros(sentence_length))).repeat(batch_size, 1)
        for i in range(sentence_length):

            # dot product attention
            attention_weight = torch.bmm(input[i].view(batch_size, 1, embedding_dimension), hidden.view(batch_size, embedding_dimension, 1))
            attention_weights[:, i] = torch.squeeze(torch.squeeze(attention_weight, 1), 1)

        # softmax to make them sum to one
        attention_weights = F.softmax(attention_weights, dim=1)

        # get weighted sum of input words in sentence
        weighted_sum = Variable(FloatTensor(torch.zeros(embedding_dimension))).repeat(batch_size, 1)
        for i in range(sentence_length):
            weighted_sum += torch.squeeze(torch.unsqueeze(attention_weights[:, i], 1) * input[i])

        # project back to original hidden layer size
        concatenation = torch.cat((torch.unsqueeze(weighted_sum, 0), hidden), 2)
        result = self.attention_layer(concatenation)

        return result


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout, SOS_index, PAD_index):
        super(Decoder, self).__init__()

        # layers
        self.lstm = nn.LSTM(input_size=200, hidden_size=hidden_size)
        self.lstm2output = nn.Linear(hidden_size, output_size)
        self.embedding = nn.Embedding(input_size, 200, padding_idx=PAD_index)
        self.dropout = nn.Dropout(p=dropout)

        # <SOS> token needed to feed decoder if no teacher forcing is used
        self.SOS = SOS_index

    def forward(self, input, hidden, context, teacher_forcing):

        if len(input) > 0:
            # get input embeddings
            input = self.embedding(torch.squeeze(input.long()))
        else:

            # get <SOS> token embedding
            batch_size = hidden.size()[1]
            input = Variable(LongTensor([self.SOS])).repeat(batch_size)
            input = self.embedding(input.long())

        input = self.dropout(input)

        if len(input.size()) == 1:
            input = torch.unsqueeze(input, 0)
        result, (hidden, context) = self.lstm(torch.unsqueeze(input, 0), (hidden, context))
        output = self.lstm2output(result)

        return output, hidden, context


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    return Variable(h.data, requires_grad=True)
