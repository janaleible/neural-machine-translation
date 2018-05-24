import collections
from typing import List

from torchtext.vocab import Vocab

import torch
from torchtext.datasets import TranslationDataset
from torchtext.data import Field

Sentence = collections.namedtuple('Sentence', 'id, english, french')
use_cuda = True if torch.cuda.is_available() else False


class ParallelData(TranslationDataset):

    def __init__(self, path: str):

        self.english_word_counts = collections.Counter()
        self.french_word_counts = collections.Counter()

        # fields
        english = Field(batch_first=True, lower=True, eos_token="<EOS>", include_lengths=True, pad_token='<PAD>')
        french = Field(include_lengths=True, batch_first=True, init_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", lower=True)

        self.english = english
        self.french = french

        super(ParallelData, self).__init__(path=path, exts=('.fr', '.en'), fields=(french, english))

    @staticmethod
    def read_file(filename: str) -> []:

        with open(filename) as infile:
            return infile.readlines()


class TestData(TranslationDataset):
    def __init__(self, path: str, english_vocabulary: Vocab, french_vocabulary: Vocab):

        self.english_vocabulary = english_vocabulary
        self.french_vocabulary = french_vocabulary



        english = Field(batch_first=True, lower=True, include_lengths=True, pad_token=english_vocabulary.stoi['<PAD>'], use_vocab=False, preprocessing=self.english2index)
        french = Field(include_lengths=True, batch_first=True, init_token=french_vocabulary.stoi['<SOS>'], eos_token=french_vocabulary.stoi['<EOS>'], pad_token=french_vocabulary.stoi['<PAD>'],
                       lower=True, use_vocab=False, preprocessing=self.french2index)

        self.english = english
        self.french = french

        super(TestData, self).__init__(path=path, exts=('.fr', '.en'), fields=(french, english))

    def english2index(self, wordlist: List[str]) -> List[int]:
        return [self.english_vocabulary.stoi[word] for word in wordlist]

    def french2index(self, wordlist: List[str]) -> List[int]:
        return [self.french_vocabulary.stoi[word] for word in wordlist]
