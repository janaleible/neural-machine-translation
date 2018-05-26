import matplotlib.pyplot as plt
import pickle
import sys

import collections
from torchtext.data import BucketIterator
from torchtext.data import interleave_keys

from predict import Predictor
from evaluation import Evaluator
from parallel_data import ParallelData, TestData

AttentionWeights = collections.namedtuple('AttentionWeights', ['weights', 'target', 'predicted'])


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
    with open('output/model_epoch1.pickle', 'rb') as file:
        model = pickle.load(file)

    batch_size = 1
    input_data = BucketIterator(
        dataset=data,
        train=True,
        batch_size=batch_size,
        # sort_key=lambda x: interleave_keys(len(x.src), len(x.trg))
    )

    training_batches = next(iter(BucketIterator(
        dataset=training_data,
        batch_size=10,
        train=True,
        sort_key=lambda x: interleave_keys(len(x.src), len(x.trg))
    )))

    predictor = Predictor(model)
    evaluator = Evaluator(training_data.english.vocab, training_data.french.vocab)

    # evaluator.add_sentences(input_data.trg[0], predictor.predict(input_data))
    for i in range(10):
        sentence = next(iter(input_data))
        predicted_sentence, attention = predictor.predict(sentence)
        evaluator.add_sentences(sentence.trg[0], predicted_sentence, attention=attention)

    attentions = len(evaluator.attention_weights)
    for i in range(attentions):
        evaluator.plot_attention(evaluator.attention_weights[i])