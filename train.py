import collections
import csv
from typing import Dict

import math

from evaluation import Evaluator
from model import NeuralMachineTranslator
from torch import optim
from parallel_data import ParallelData, TestData
from helpers import get_validation_metrics
import pickle
import torch
from torchtext.data import BucketIterator, interleave_keys
from time import time

Metrics = collections.namedtuple('Metrics', ['BLEU', 'TER', 'loss'])
use_cuda = torch.cuda.is_available()


def train(batch, model, use_teacher_forcing):

    output, loss, _ = model(batch, teacher_forcing=use_teacher_forcing, get_loss=True)

    return output, loss


def train_epochs(
    training_data: ParallelData,
    embedding_dimension: int,
    n_epochs: int,
    batch_size: int,
    max_sentence_length: int,
    evaluator: Evaluator,
    validation_evaluator: Evaluator,
    dropout=0.3,
    learning_rate=0.01,
    max_iterations_per_epoch=math.inf,
    teacher_forcing=False
) -> Dict[int, Metrics]:

    n_english = len(training_data.english.vocab)
    n_french = len(training_data.french.vocab)

    # iterators
    train_iterator = BucketIterator(dataset=training_data, batch_size=batch_size,
                                    sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=True)

    validation_data = TestData("data/BPE/valid/val.BPE", training_data.english.vocab, training_data.french.vocab)
    validation_iterations = (len(validation_data) // batch_size) + 1
    validation_iterator = BucketIterator(dataset=validation_data, batch_size=batch_size,
                                         sort_key=lambda x: interleave_keys(len(x.src), len(x.trg)), train=False)

    iterations_per_epoch = min(max_iterations_per_epoch, (len(training_data) // batch_size) + 1)

    model = NeuralMachineTranslator(
        embedding_dimension,
        n_french,
        max_sentence_length,
        dropout,
        n_english,
        n_english,
        2*embedding_dimension,
        batch_size,
        training_data.english.vocab.stoi['<EOS>'],
        training_data.english.vocab.stoi['<SOS>'],
        training_data.english.vocab.stoi['<PAD>'],
        max_prediction_length=max_sentence_length
    )

    model.train()

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    print("Parameters to train: ")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print()

    metrics = {}
    validation_metrics = {}
    training_metrics = {}

    print("Start training..")
    for epoch in range(1, n_epochs + 1):

        epoch_loss = 0
        iteration_loss = 0

        start_time = time()
        for iteration in range(iterations_per_epoch):

            # set gradients to zero
            optimizer.zero_grad()
            model.zero_grad()

            # get next batch
            batch = next(iter(train_iterator))

            # forward pass
            prediction, loss = train(batch, model, teacher_forcing)

            # # backward pass final step without retaining graph
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)

            # update parameters final step
            optimizer.step()

            # save losses and add predicted sentences to evaluator
            epoch_loss += loss.item()
            iteration_loss += loss.item()
            evaluator.add_sentences(batch.trg[0], prediction, model.EOS)

            if iteration > 1 and iteration % 200 == 0:
                current_time = (time() - start_time) / 200
                print('batch {}/{}'.format(iteration, iterations_per_epoch))
                print('average loss per batch: {:5.3}'.format(iteration_loss / 200))
                print("time per batch {:3}".format(current_time))
                iteration_loss = 0
                start_time = time()

        # save evaluation metrics
        metrics[epoch] = Metrics(evaluator.bleu(), evaluator.ter(), float(epoch_loss))
        evaluator.write_to_file('output/predictions_epoch{}'.format(epoch))

        # clear sentences from evaluator
        evaluator.clear_sentences()

        print(
            'Epoch {}: training metrics: loss {:.3}, BLEU {:.3}, TER {:.3}, LR {:.3}'.format(
                epoch, float(metrics[epoch].loss), float(metrics[epoch].BLEU), float(metrics[epoch].TER), float(learning_rate)
            )
        )

        print("Getting validation metrics..")

        validation_metrics[epoch], training_metrics[epoch] = get_validation_metrics(
            model,
            validation_iterations,
            evaluator,
            validation_evaluator,
            train_iterator,
            validation_iterator
        )

        # clear sentences out of evaluators
        evaluator.clear_sentences()
        validation_evaluator.clear_sentences()
        print(
            'Epoch {}: validation metrics: BLEU {:.3}, TER {:.3}'.format(
                epoch, float(validation_metrics[epoch].BLEU), float(validation_metrics[epoch].TER)
            )
        )

        if epoch > 1 and metrics[epoch].loss > metrics[epoch - 1].loss:
            learning_rate /= 4
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        with open('training_progress.csv', 'w') as file:
            filewriter = csv.writer(file)
            filewriter.writerow(['Epoch', 'loss', 'BLEU', 'TER'])
            for epoch, metric in metrics.items():
                filewriter.writerow([epoch, metric.loss, metric.BLEU, metric.TER])

        with open('validation_progress.csv', 'w') as file:
            filewriter = csv.writer(file)
            filewriter.writerow(['Epoch', 'training BLEU', 'valid BLEU', 'training TER', 'valid TER'])
            for epoch, metric in validation_metrics.items():
                filewriter.writerow([epoch, training_metrics[epoch].BLEU, validation_metrics[epoch].BLEU,
                                     training_metrics[epoch].TER, validation_metrics[epoch].TER])

        with open('output/model_epoch{}.pickle'.format(epoch), 'wb') as file:
            pickle.dump(model, file)

    return metrics


# noinspection PyPackageRequirements
if __name__ == "__main__":

    data_path = "data/"

    # paths to data
    train_path = data_path + "BPE/train/train.BPE"
    validation_path = data_path + "BPE/valid/val.BPE"
    test_path = data_path + "BPE/test/test.BPE"

    # locations to save data
    filename_train = 'pickles/train_data.pickle'
    filename_valid = 'pickles/validation_data.pickle'
    filename_test = 'pickles/test_data.pickle'

    # hyper parameters
    embedding_dimension = 100
    batch_size = 32
    epochs = 50
    max_sentence_length = 30
    max_iterations_per_epoch = 30
    dropout = 0
    initial_learning_rate = 0.2
    teacher_forcing = True

    # get data
    training_data = ParallelData(train_path)

    # build vocabulary
    training_data.french.build_vocab(training_data, max_size=80000)
    training_data.english.build_vocab(training_data, max_size=40000)

    # save vocabulary
    torch.save(training_data.french.vocab, 'pickles/french_vocab.txt')
    torch.save(training_data.english.vocab, 'pickles/english_vocab.txt')

    # initialize evaluators
    evaluator = Evaluator(training_data.english.vocab, training_data.french.vocab)
    validation_evaluator = Evaluator(training_data.english.vocab, training_data.french.vocab)

    # train
    train_epochs(
        training_data,
        embedding_dimension,
        epochs,
        batch_size,
        max_sentence_length,
        evaluator,
        validation_evaluator,
        dropout,
        initial_learning_rate,
        # max_iterations_per_epoch=max_iterations_per_epoch,
        teacher_forcing=teacher_forcing
    )
