import collections
import os
from typing import List
from moses_tokenizer import MosesTokenizer
import json

import pickle

Sentence = collections.namedtuple('Sentence', 'id, english, french')


class ParallelData(object):
    """
    Read in parallel dataset from HLT-NAACL
    http://www.cs.unt.edu/~rada/wpt
    """

    def __init__(self, path: str, tokenize: bool):

        self.english_word_counts = collections.Counter()
        self.french_word_counts = collections.Counter()

        self.tokenizer = MosesTokenizer()

        # file names
        if tokenize:
            train_file_e = path + "train.en"
            train_file_f = path + "train.fr"
            validation_file_e = path + "val.en"
            validation_file_f = path + "val.fr"
            test_file_e = path + "test_2017_flickr.en"
            test_file_f = path + "test_2017_flickr.fr"
            data_files = [train_file_e, train_file_f, validation_file_e, validation_file_f, test_file_e, test_file_f]
            tokenized_files = ["train.token.en", "train.token.fr", "val.token.en", "val.token.fr", "test.token.en", "test.token.fr"]
            for infile, outfile in zip(data_files, tokenized_files):
                self.tokenize_and_lowercase(infile, outfile)

        # BPE filenames
        train_file_e = path + "BPE/train/train.BPE.en"
        train_file_f = path + "BPE/train/train.BPE.fr"
        validation_file_e = path + "BPE/valid/val.BPE.en"
        validation_file_f = path + "BPE/valid/val.BPE.fr"
        test_file_e = path + "BPE/test/test.BPE.en"
        test_file_f = path + "BPE/test/test.BPE.fr"
        dictionary_file_e = path + "BPE/train.BPE.en.json"
        dictionary_file_f = path + "BPE/train.BPE.fr.json"

        # read data
        self.train_data = self.read_sentence_pairs(train_file_e, train_file_f)
        self.validation_data = self.read_sentence_pairs(validation_file_e, validation_file_f)
        self.test_data = self.read_sentence_pairs(test_file_e, test_file_f)

        # read dictionaries
        self.english_vocabulary = self.read_dictionary(dictionary_file_e)
        self.french_vocabulary = self.read_dictionary(dictionary_file_f)

    def tokenize_and_lowercase(self, infile: str, outfile: str):

        sentences = self.read_file(infile)
        with open(outfile, "w") as file:
            for sentence in sentences:
                tokenized = " ".join(self.tokenizer.tokenize(sentence.lower()))
                file.write(tokenized)
                file.write("\n")

    def read_dictionary(self, file: str) -> dict:

        with open(file, "r") as infile:
            dictionary = json.load(infile)

        return dictionary

    def read_sentence_pairs(self, file_e: str, file_f: str) -> (List[Sentence], int):

        data = []
        e_sentences = self.read_file(file_e)
        f_sentences = self.read_file(file_f)
        next_id = 0
        end_of_sentence_token = "<EOS>"

        for english, french in zip(e_sentences, f_sentences):
            sentence = Sentence(
                next_id,
                english.split() + [end_of_sentence_token],
                french.split() + [end_of_sentence_token]
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

        data = ParallelData('data/', False)

        with open(filename, 'wb') as file:
            pickle.dump(data, file)