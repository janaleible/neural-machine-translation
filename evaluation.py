import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor
from torchtext.vocab import Vocab


class Evaluator:

    def __init__(self, french_vocabulary: Vocab, english_vocabulary: Vocab):

        self.french_vocabulary = french_vocabulary
        self.english_vocabulary = english_vocabulary

        self.input_sentences = []
        self.translated_sentences = []

    def add_sentences(self, input_batch: Tensor, predicted_batch: np.ndarray):

        batch_size = input_batch.size()[0]

        for sentence in range(batch_size):
            self.input_sentences.append(self.index2Text(input_batch[sentence, :], self.french_vocabulary))
            self.translated_sentences.append(self.index2Text(predicted_batch[sentence, :], self.english_vocabulary))

    def index2Text(self, sentence_indices: [], vocabulary: Vocab) -> []:
        sentence_bpe = [vocabulary.itos[int(index)] for index in filter(lambda index: int(index) is not vocabulary.stoi['<PAD>'], sentence_indices)]

        parsed = []
        subword_stack = ''
        stack_mode = False
        for subword in sentence_bpe:
            if subword.endswith('@@'):
                stack_mode = True
                subword_stack += subword[:-2]
            else:
                if stack_mode:
                    subword_stack += subword
                    parsed.append(subword_stack)
                    subword_stack = ''
                    stack_mode = False
                else:
                    parsed.append(subword)

        return parsed

    def bleu(self) -> float:
        return corpus_bleu(self.input_sentences, self.translated_sentences)

    def meteor(self) -> float:
        raise NotImplementedError

    def ter(self) -> float:
        raise NotImplementedError
