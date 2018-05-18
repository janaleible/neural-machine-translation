import collections
import os
from typing import List
from lib.moses.moses import MosesTokenizer
import torch
from torchtext.datasets import TranslationDataset
from torchtext.data import BucketIterator, Field, interleave_keys

import pickle

Sentence = collections.namedtuple('Sentence', 'id, english, french')
use_cuda = True if torch.cuda.is_available() else False


class ParallelData(TranslationDataset):

    def __init__(self, path: str):

        self.english_word_counts = collections.Counter()
        self.french_word_counts = collections.Counter()

        # fields
        english = Field(batch_first=True, lower=True, include_lengths=True, pad_token='<PAD>')
        french = Field(include_lengths=True, batch_first=True, init_token="<SOS>", eos_token="<EOS>", pad_token="<PAD>", lower=True)

        self.english = english
        self.french = french

        super(ParallelData, self).__init__(path=path, exts=('.fr', '.en'), fields=(french, english))

    def tokenize_and_lowercase(self, infile: str, outfile: str):

        sentences = self.read_file(infile)
        with open(outfile, "w") as file:
            for sentence in sentences:
                tokenized = " ".join(self.tokenizer.tokenize(sentence.lower()))
                file.write(tokenized)
                file.write("\n")

    def read_sentence_pairs(self, file_e: str, file_f: str) -> (List[Sentence], int):

        e_sentences = self.read_file(file_e)
        f_sentences = self.read_file(file_f)
        next_id = 0

        sentence_dict = {"english": [], "french": []}

        for english, french in zip(e_sentences, f_sentences):
            # sentence = Sentence(
            #     next_id,
            #     english.split() + [end_of_sentence_token],
            #     french.split() + [end_of_sentence_token]
            # )
            # sentence_dict["id"] = next_id
            sentence_dict["english"] = english.split()
            sentence_dict["french"] = french.split()

            for word in french.split():
                self.french_word_counts[word] += 1

            for word in english.split():
                self.english_word_counts[word] += 1

            next_id += 1
            yield sentence_dict


    @staticmethod
    def read_file(filename: str) -> []:

        with open(filename) as infile:
            return infile.readlines()


if __name__ == '__main__':

    filename = 'pickles/data.pickle'

    if not os.path.isfile(filename):

        data = ParallelData('data/', False)

        with open(filename, 'wb') as file:
            pickle.dump(data, file)