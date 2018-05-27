import pickle
from typing import Tuple

import sys

import numpy as np
from torchtext.data import Batch
from torchtext.data import BucketIterator
from torchtext.data import interleave_keys

from evaluation import Evaluator
from model import NeuralMachineTranslator
from parallel_data import ParallelData, TestData


class Predictor:

    def __init__(self, model: NeuralMachineTranslator):
        self.model = model

    def predict(self, data: Batch) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        prediction, _, attention = self.model(data, get_loss=False, teacher_forcing=False)
        return prediction, attention


if __name__ == "__main__":

    dataset = sys.argv[1]

    training_data = ParallelData('data/BPE/train/train.BPE')
    training_data.french.build_vocab(training_data, max_size=80000)
    training_data.english.build_vocab(training_data, max_size=40000)
    eos_token = training_data.english.vocab.stoi["<EOS>"]

    if dataset == 'validation':
        data = TestData("data/BPE/valid/val.BPE", training_data.english.vocab, training_data.french.vocab)
    elif dataset == 'test':
        data = TestData("data/BPE/test/test.BPE", training_data.english.vocab, training_data.french.vocab)
    else:
        raise ValueError('Unknown dataset, pick one of validation/test')
    with open('output/model_epoch3.pickle', 'rb') as file:
        model = pickle.load(file)

    batch_size = 1
    input_data = BucketIterator(
        dataset=data,
        train=True,
        batch_size=batch_size
    )

    training_batches = next(iter(BucketIterator(
        dataset=training_data,
        batch_size=10,
        train=True,
        sort_key=lambda x: interleave_keys(len(x.src), len(x.trg))
    )))
    with open('test.txt', 'w') as file:

        predictor = Predictor(model)
        evaluator = Evaluator(training_data.english.vocab, training_data.french.vocab)

        # evaluator.add_sentences(input_data.trg[0], predictor.predict(input_data))
        for i in range((len(data) // batch_size) + 1):
            sentence = next(iter(input_data))
            predicted_sentence, _ = predictor.predict(sentence)
            evaluator.add_sentences(sentence.trg[0], predicted_sentence, eos_token)
        #
        # for i in range((len(data) // batch_size) + 1):
        #     sentence = next(iter(input_data))
        #     src, trg = evaluator.convert_sentences(sentence)
        #     file.write(' '.join(src) + '\n')
        #     file.write(' '.join(trg) + '\n')
        #     file.write('\n')

        evaluator.write_to_file("output/validation_predictions_epoch{}".format(3))
        print(evaluator.bleu_test("output/validation_predictions_epoch3.hyp", "output/validation_predictions_epoch3.ref"))
        print('bleu:', evaluator.bleu())
        print('ter: ', evaluator.ter())
        print('')
