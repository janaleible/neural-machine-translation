import collections
from model import NeuralMachineTranslator
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


def train(batch, model):

    output, loss = model(batch)

    # TODO: update parameters?
    return loss


def train_epochs(path, embedding_dimension, n_epochs, batch_size, max_sentence_length, dropout=0.3, learning_rate=0.01):

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

    model = NeuralMachineTranslator(embedding_dimension, n_french, max_sentence_length, dropout,
                 1, n_english, 2*embedding_dimension)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):

        epoch_loss = 0
        for iteration in range(iterations_per_epoch):

            # get next batch
            batch = next(iter(train_iterator))
            loss = train(batch, model)
            loss.backward()
            optimizer.step()
            epoch_loss += loss


if __name__ == "__main__":

    path_to_data = "data/"

    # hyperparameters
    embedding_dimension = 50
    batch_size = 32
    epochs = 1
    max_sentence_length = 150

    # train
    train_epochs(path_to_data, embedding_dimension, epochs, batch_size, max_sentence_length)
