import collections
import os
from typing import List
from moses_tokenizer import MosesTokenizer

import pickle

Sentence = collections.namedtuple('Sentence', 'id, english, french')


class ParallelData(object):
    """
    Read in parallel dataset from HLT-NAACL
    http://www.cs.unt.edu/~rada/wpt
    """

    def __init__(self, path: str):

        self.english_word_counts = collections.Counter()
        self.french_word_counts = collections.Counter()

        self.training_vocabulary_english = set()
        self.training_vocabulary_french = set()

        self.tokenizer = MosesTokenizer()

        # file names
        train_file_e = path + "train.en"
        train_file_f = path + "train.fr"
        validation_file_e = path + "val.en"
        validation_file_f = path + "val.fr"
        test_file_e = path + "test_2017_flickr.en"
        test_file_f = path + "test_2017_flickr.fr"

        # read data
        self.train_data = self.read_sentence_pairs(train_file_e, train_file_f)
        self.validation_data = self.read_sentence_pairs(validation_file_e, validation_file_f)
        self.test_data = self.read_sentence_pairs(test_file_e, test_file_f)

    def read_sentence_pairs(self, file_e: str, file_f: str) -> (List[Sentence], int):

        data = []
        e_sentences = self.read_file(file_e)
        f_sentences = self.read_file(file_f)
        next_id = 0
        end_of_sentence_token = "<EOS>"

        for english, french in zip(e_sentences, f_sentences):
            sentence = Sentence(
                next_id,
                self.tokenizer.tokenize(english.lower(), escape=False) + [end_of_sentence_token],
                self.tokenizer.tokenize(french.lower(), escape=False) + [end_of_sentence_token]
            )
            data.append(sentence)

            for word in sentence.french:
                self.french_word_counts[word] += 1

            for word in sentence.english:
                self.english_word_counts[word] += 1

            next_id += 1
        return data


    @staticmethod
    def read_file(filename: str) -> []:

        with open(filename) as infile:
            return infile.readlines()


if __name__ == '__main__':

    filename = 'pickles/data.pickle'

    if not os.path.isfile(filename):

        data = ParallelData('data/')

        with open(filename, 'wb') as file:
            pickle.dump(data, file)