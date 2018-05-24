from typing import Tuple

from torchtext.data import BucketIterator

from evaluation import Evaluator
from model import NeuralMachineTranslator
from predict import Predictor
import collections

Metrics = collections.namedtuple('Metrics', ['BLEU', 'TER', 'loss'])


def get_validation_metrics(model: NeuralMachineTranslator,
                           iterations: int,
                           training_evaluator: Evaluator,
                           validation_evaluator: Evaluator,
                           training_iterator: BucketIterator,
                           validation_iterator: BucketIterator) -> Tuple[Metrics, Metrics]:
    # get predictor
    predictor = Predictor(model)

    validation_evaluator.clear_sentences()
    # loop over validation sentences and add predictions to evaluator
    for i in range(iterations):
        validation_batch = next(iter(validation_iterator))
        validation_evaluator.add_sentences(validation_batch.trg[0], predictor.predict(validation_batch), model.EOS)

    # get validation metrics
    validation_metrics = Metrics(validation_evaluator.bleu(), validation_evaluator.ter(), 0)

    training_evaluator.clear_sentences()
    # get 50 batches from training data and add predictions to evaluator
    for i in range(50):
        batch = next(iter(training_iterator))
        training_evaluator.add_sentences(batch.trg[0], predictor.predict(batch), model.EOS)

    # get training metrics
    training_metrics = Metrics(training_evaluator.bleu(), training_evaluator.ter(), 0)

    return validation_metrics, training_metrics
