import collections
from model import PositionalEncoder, Attention, Decoder
import logging
from torch import optim
from data import ParallelData
import pickle

from torch import FloatTensor
import torch
from torch.autograd import Variable
from torchtext.data import BucketIterator, Field, interleave_keys
from torch import nn

logger = logging.getLogger(__name__)
Sentence = collections.namedtuple('Sentence', 'id, english, french')
use_cuda = True if torch.cuda.is_available() else False


# def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)
#
#     loss = 0
#
#     encoder_outputs = torch.zeros(1, encoder.embedding_dimension, cuda=use_cuda)
#     for i in range(1, input_length + 1):  # pos runs from 1 or 0?
#         encoder_output, encoder_hidden = encoder(input_tensor[i - 1], i)
#         encoder_outputs += encoder_output  # sum encoder outputs?
#
#     decoder_input = torch.tensor(["<SOS>"], cuda=use_cuda)  # Start of Sentence token?
#
#     decoder_hidden = decoder.hidden  # needs to be initialized
#
#     # Teacher forcing: Feed the target as the next input
#     for i in range(target_length):
#         decoder_output, decoder_hidden, decoder_attention = decoder(
#             decoder_input, decoder_hidden, encoder_outputs)
#         loss += criterion(decoder_output, target_tensor[o])
#         decoder_input = target_tensor[i]  # Teacher forcing
#
#     loss.backward()
#
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.item() / target_length


def dump_data(f, content):
    with open(f, 'wb') as file:
        pickle.dump(content, file)


def load_data(f):
    with open(f, 'rb') as file:
        data = pickle.load(file)

    return data


def train_iterations(path, dimension, embedding_dimension, n_iterations, batch_size, dropout=0.3, learning_rate=0.01):

    logger.info("Start training..")

    # paths to data
    train_path = path + "BPE/train/train.BPE"
    validation_path = path + "BPE/valid/val.BPE"
    test_path = path + "BPE/test/test.BPE"

    # TODO: save data to pickles?
    # locations to save data
    filename_train = 'pickles/train_data.pickle'
    filename_valid = 'pickles/validation_data.pickle'
    filename_test = 'pickles/test_data.pickle'

    # get data
    training_data = ParallelData(train_path)
    validation_data = ParallelData(validation_path)
    test_data = ParallelData(test_path)

    # build vocabulary
    # TODO: I think it automatically unks..
    training_data.french.build_vocab(training_data, max_size=80000)
    training_data.english.build_vocab(training_data, max_size=40000)

    # get iterators
    n_english = len(training_data.english.vocab)
    n_french = len(training_data.french.vocab)


    # iterators
    train_iterator = BucketIterator(dataset=training_data, batch_size=batch_size,
                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=True)
    validation_iterator = BucketIterator(dataset=validation_data, batch_size=32,
                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=False)
    test_iterator = BucketIterator(dataset=test_data, batch_size=32,
                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=False)

    iterations_per_epoch = (len(training_data) // batch_size) + 1

    encoder = PositionalEncoder(dimension, embedding_dimension, n_french, 100, dropout)
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    # decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    output_softmax = nn.Softmax(dim=2)

    attention = Attention(embedding_dimension)
    criterion = nn.NLLLoss(size_average=False, reduce=False)

    for iteration in range(1, n_iterations + 1):

        batch = next(iter(train_iterator))

        input_sentences = batch.src[0]
        input_lengths = batch.src[1]

        target_sentences = batch.trg[0]
        target_lengths = batch.trg[1]

        french_sentence_length = input_sentences.size()[1]
        english_sentence_length = target_sentences.size()[1]

        mask = torch.zeros((batch_size, english_sentence_length))
        for sentence in range(batch_size):
            sentence_mask = [0] * int(target_lengths[sentence]) + [1] * (english_sentence_length - int(target_lengths[sentence]))
            mask[sentence, :] = torch.LongTensor(sentence_mask)


        batch_size, time_size = input_sentences.size()
        encoder_outputs = []
        average_embedding = Variable(FloatTensor(torch.zeros(2 * embedding_dimension))).repeat(batch_size, 1)
        for time in range(time_size):
            positional_embedding = encoder(input_sentences[:, time], time + 1)
            encoder_outputs.append(positional_embedding)
            average_embedding += positional_embedding

        average_embedding /= time_size


        decoder = Decoder(1, 2*embedding_dimension, n_english)
        hidden = torch.unsqueeze(average_embedding, 0)

        context = torch.randn(hidden.size())
        loss = 0
        for time in range(english_sentence_length):
            hidden = attention(encoder_outputs, hidden)
            output, hidden, context = decoder(
                torch.unsqueeze(torch.unsqueeze(target_sentences[:, time], 0), 2).float(),
                hidden,
                context
            )

            output = output_softmax(output)

            batch_loss = criterion(torch.squeeze(output), target_sentences[:, time])
            batch_loss.masked_fill_(mask[:, time].byte(), 0.)
            loss += batch_loss.sum() / batch_size

        loss.backward()




if __name__ == "__main__":
    dimension = 50
    embedding_dimension = 50
    batch_size = 32
    iterations = 1
    path_to_data = "data/"
    train_iterations(path_to_data, dimension, embedding_dimension, iterations, batch_size)
