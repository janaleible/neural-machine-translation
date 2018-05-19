import numpy as np
import pyter
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor
from torchtext.vocab import Vocab


class Evaluator:

    def __init__(self, english_vocabulary: Vocab):

        self.english_vocabulary = english_vocabulary

        self.target_sentences = []
        self.translated_sentences = []

    def add_sentences(self, target_batch: Tensor, predicted_batch: np.ndarray):

        batch_size = target_batch.size()[0]

        for sentence in range(batch_size):
            self.target_sentences.append(self.index2Text(target_batch[sentence, :]))
            self.translated_sentences.append(self.index2Text(predicted_batch[sentence, :]))

    def index2Text(self, sentence_indices: []) -> []:
        sentence_bpe = [self.english_vocabulary.itos[int(index)] for index in filter(lambda index: int(index) is not self.english_vocabulary.stoi['<PAD>'], sentence_indices)]

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
        return corpus_bleu(self.target_sentences, self.translated_sentences)

    def meteor(self) -> float:
        raise NotImplementedError

    def ter(self) -> float:
        total_ter = 0
        for translation, target in zip(self.translated_sentences, self.target_sentences):
            pyter.ter(translation, target)
        return total_ter / len(self.translated_sentences)
