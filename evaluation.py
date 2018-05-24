import numpy as np
import os
# import pyter
from lib.pyter import ter
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor
from torchtext.vocab import Vocab


class Evaluator:

    def __init__(self, english_vocabulary: Vocab):

        self.english_vocabulary = english_vocabulary

        self.target_sentences = []
        self.translated_sentences = []

    def add_sentences(self, target_batch: Tensor, predicted_batch: np.ndarray, eos_token=2):

        batch_size = target_batch.size()[0]

        for sentence in range(batch_size):

            indices = np.where(predicted_batch[sentence, :] == float(eos_token))[0]
            if len(indices) >= 1:
                eos_index = int(np.where(predicted_batch[sentence, :] == float(eos_token))[0][0])
            else:
                eos_index = -1
            sentence_without_padding = predicted_batch[sentence, :eos_index]
            self.target_sentences.append(self.index2Text(target_batch[sentence, :]))
            self.translated_sentences.append(self.index2Text(sentence_without_padding))

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
        return corpus_bleu([[target_sentece] for target_sentece in self.target_sentences], self.translated_sentences)

    def meteor(self) -> float:
        raise NotImplementedError

    def ter(self) -> float:
        total_ter = 0
        for translation, target in zip(self.translated_sentences, self.target_sentences):
            total_ter += ter(translation, target)

        return total_ter / len(self.translated_sentences)

    def write_to_file(self, path):

        directory = path.split('/')[:-1]
        path_to_file = os.path.join(*directory)

        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        with open(path + '.ref', 'w') as file:
            for sentence in self.target_sentences:
                file.write(' '.join(sentence) + '\n')

        with open(path + '.hyp', 'w') as file:
            for sentence in self.translated_sentences:
                file.write(' '.join(sentence) + '\n')

    def clear_sentences(self):
        self.translated_sentences = []
        self.target_sentences = []
