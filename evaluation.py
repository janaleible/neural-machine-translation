import numpy as np
import os
# import pyter
from lib.pyter import ter
from nltk.translate.bleu_score import corpus_bleu
from torch import Tensor
from torchtext.vocab import Vocab
import collections
import matplotlib.pyplot as plt
import subprocess
from six.moves import urllib
import re

AttentionWeights = collections.namedtuple('AttentionWeights', ['weights', 'target', 'predicted'])


class Evaluator:

    def __init__(self, english_vocabulary: Vocab, french_vocabulary: Vocab):

        self.english_vocabulary = english_vocabulary
        self.french_vocabulary = french_vocabulary
        self.target_sentences = []
        self.translated_sentences = []
        self.attention_weights = []

    def add_sentences(self, target_batch: Tensor, predicted_batch: np.ndarray, eos_token=2, attention=None):

        batch_size = target_batch.size()[0]

        for sentence in range(batch_size):
            indices = np.where(predicted_batch[sentence, :] == float(eos_token))[0]
            if len(indices) >= 1:
                eos_index = int(np.where(predicted_batch[sentence, :] == float(eos_token))[0][0])
            else:
                eos_index = -1
            sentence_without_padding = predicted_batch[sentence, :eos_index]
            translated_sentence_text = self.index2Text(sentence_without_padding, "english")
            target_sentence_text = self.index2Text(target_batch[sentence, :], "english")
            if attention is not None:
                input_sentence = self.index2Text(attention.input.detach().numpy()[0], "french")
                self.attention_weights.append(AttentionWeights(np.squeeze(attention.weights.detach().numpy()),
                                                               input_sentence,
                                                               translated_sentence_text))
            self.target_sentences.append(target_sentence_text)
            self.translated_sentences.append(translated_sentence_text)

    def index2Text(self, sentence_indices: [], language: str) -> []:

        if language == "french":
            vocabulary = self.french_vocabulary
        else:
            vocabulary = self.english_vocabulary
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
        return corpus_bleu([[target_sentece] for target_sentece in self.target_sentences], self.translated_sentences)

    def bleu_test(self, hypothesis_path: str, reference_path: str) -> float:

        multi_bleu_path, _ = urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
            "master/scripts/generic/multi-bleu.perl")

        os.chmod(multi_bleu_path, 0o755)
        multi_bleu_path = [multi_bleu_path]

        multi_bleu_path += [reference_path]
        print(multi_bleu_path)
        with open(hypothesis_path, "r") as read_pred:
            bleu_out = subprocess.check_output(multi_bleu_path, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)

        return bleu_score

    def meteor(self) -> float:
        raise NotImplementedError

    def ter(self) -> float:
        total_ter = 0
        for translation, target in zip(self.translated_sentences, self.target_sentences):
            total_ter += ter(translation, target)

        return total_ter / len(self.translated_sentences)

    def convert_sentences(self, sentence):
        src_french = self.index2Text(sentence.src[0].detach().numpy()[0], "french")
        trg_english = self.index2Text(sentence.trg[0].detach().numpy()[0], "english")
        return src_french, trg_english

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

    def plot_attention(self, attention: AttentionWeights):
        """
        Plots the scatterplot of each measurement against each other measurement
        """
        predicted_sentence = attention.predicted
        target_sentence = attention.target
        weights = attention.weights
        n_predicted = len(predicted_sentence)
        n_target = len(target_sentence)
        weight_matrix = np.zeros((n_predicted, n_target))
        for i in range(len(predicted_sentence)):
            for j in range(len(target_sentence)):
                weight_matrix[i][j] = weights[i][j]
        plt.imshow(weight_matrix)
        plt.xticks(range(n_target), tuple(target_sentence))
        plt.yticks(range(n_predicted), tuple(predicted_sentence))
        plt.show()

    def visualize_attention(self, path):

        directory = path.split('/')[:-1]
        path_to_file = os.path.join(*directory)

        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        with open(path + '.ref', 'w') as file:
            for sentence in self.target_sentences:
                file.write(' '.join(sentence) + '\n')

    def clear_sentences(self):
        self.translated_sentences = []
        self.target_sentences = []

    def clear_attention_weights(self):
        self.attention_weights = []