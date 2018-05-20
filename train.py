import collections
import csv
from typing import Dict

import math

from evaluation import Evaluator
from model import NeuralMachineTranslator
import logging
from torch import optim
from parallel_data import ParallelData
import pickle
import torch
from torchtext.data import BucketIterator, interleave_keys

Metrics = collections.namedtuple('Metrics', ['BLEU', 'TER', 'loss'])

logger = logging.getLogger(__name__)
Sentence = collections.namedtuple('Sentence', 'id, english, french')
use_cuda = torch.cuda.is_available()


def dump_data(f, content):
    with open(f, 'wb') as file:
        pickle.dump(content, file)


def load_data(f):
    with open(f, 'rb') as file:
        data = pickle.load(file)

    return data


def train(batch, model):
    model.zero_grad()
    output, loss = model(batch)
    return output, loss


def train_epochs(
    training_data: ParallelData,
    embedding_dimension: int,
    n_epochs: int,
    batch_size: int,
    max_sentence_length: int,
    evaluator: Evaluator,
    dropout=0.3,
    learning_rate=0.05,
    max_iterations_per_epoch=math.inf
) -> Dict[int, Metrics]:

    logger.info("Start training..")
    print("Start training..")

    # get iterators
    n_english = len(training_data.english.vocab)
    n_french = len(training_data.french.vocab)

    # iterators
    train_iterator = BucketIterator(dataset=training_data, batch_size=batch_size,
                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=True)
    # validation_iterator = BucketIterator(dataset=validation_data, batch_size=32,
    #                                 sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=False)
    # test_iterator = BucketIterator(dataset=test_data, batch_size=32,
    #                                 sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=False)

    iterations_per_epoch = min(max_iterations_per_epoch, (len(training_data) // batch_size) + 1)

    model = NeuralMachineTranslator(embedding_dimension, n_french, max_sentence_length, dropout,
                 n_english, n_english, 2*embedding_dimension, batch_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    metrics = {}

    for epoch in range(1, n_epochs + 1):

        # print('Epoch {}'.format(epoch))

        epoch_loss = 0
        for iteration in range(iterations_per_epoch):

            # get next batch
            optimizer.zero_grad()
            batch = next(iter(train_iterator))
            prediction, loss = train(batch, model)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
            evaluator.add_sentences(batch.trg[0], prediction)

            if iteration % 100 == 0:
                print('batch {}/{}'.format(iteration, iterations_per_epoch))
                print(loss)

        metrics[epoch] = Metrics(evaluator.bleu(), 0, epoch_loss)
        evaluator.write_to_file('output/predictions_epoch{}.pred'.format(epoch))

        print(
            'Epoch {}: loss {:.3}, BLEU {:.3}, TER {:.3}'.format(
                epoch, float(metrics[epoch].loss), float(metrics[epoch].BLEU), float(metrics[epoch].TER)
            )
        )

        with open('training_progress.csv', 'w') as file:
            filewriter = csv.writer(file)
            filewriter.writerow(['Epoch', 'loss', 'BLEU', 'TER'])
            for epoch, metric in metrics.items():
                filewriter.writerow([epoch, metric.loss, metric.BLEU, metric.TER])

        with open('output/model_epoch{}.pickle'.format(epoch), 'wb') as file:
            pickle.dump(model, file)

    return metrics


if __name__ == "__main__":

    data_path = "data/"

    # paths to data
    train_path = data_path + "BPE/train/train.BPE"
    validation_path = data_path + "BPE/valid/val.BPE"
    test_path = data_path + "BPE/test/test.BPE"

    # TODO: save data to pickles?
    # locations to save data
    filename_train = 'pickles/train_data.pickle'
    filename_valid = 'pickles/validation_data.pickle'
    filename_test = 'pickles/test_data.pickle'

    # get data
    training_data = ParallelData(train_path)
    # validation_data = ParallelData(validation_path)
    # test_data = ParallelData(test_path)

    # build vocabulary
    # TODO: I think it automatically unks..
    training_data.french.build_vocab(training_data, max_size=80000)
    training_data.english.build_vocab(training_data, max_size=40000)

    # hyperparameters
    embedding_dimension = 100
    batch_size = 32
    epochs = 10
    max_sentence_length = 150
    max_iterations_per_epoch = 10

    evaluator = Evaluator(training_data.english.vocab)

    # train
    train_epochs(
        training_data,
        embedding_dimension,
        epochs,
        batch_size,
        max_sentence_length,
        evaluator,
        # max_iterations_per_epoch=max_iterations_per_epoch
    )
