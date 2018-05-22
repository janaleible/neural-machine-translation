from typing import List

import pickle

import sys

import numpy as np
import torch
from torchtext.data import Batch
from torchtext.data import BucketIterator
from torchtext.data import interleave_keys

from evaluation import Evaluator
from model import NeuralMachineTranslator
from parallel_data import ParallelData, TestData


class Predictor:

    def __init__(self, model: NeuralMachineTranslator):
        self.model = model

    def predict(self, data: Batch) -> np.ndarray:
        return self.model(data)


if __name__ == "__main__":

    dataset = sys.argv[1]




    training_data = ParallelData('data/BPE/train/train.BPE')
    training_data.french.build_vocab(training_data, max_size=80000)
    training_data.english.build_vocab(training_data, max_size=40000)

    if dataset == 'validation':
        data = TestData("data/BPE/valid/val.BPE", training_data.english.vocab, training_data.french.vocab)
    elif dataset == 'test':
        data = TestData("data/BPE/test/test.BPE", training_data.english.vocab, training_data.french.vocab)
    else:
        raise ValueError('Unknown dataset, pick one of validation/test')
    with open('output/model_epoch1.pickle', 'rb') as file:
        model = pickle.load(file)

    model.EOS = training_data.english.vocab.stoi['<EOS>']
    model.max_prediction_length = 50
    model.start = True

    input_data = next(iter(BucketIterator(
        dataset=data,
        batch_size=len(data),
        train=True,
        sort_key=lambda x: interleave_keys(len(x.src), len(x.trg))
    )))

    predictor = Predictor(model)
    evaluator = Evaluator(training_data.english.vocab)

    evaluator.add_sentences(input_data.trg[0], predictor.predict(input_data))

    print('')