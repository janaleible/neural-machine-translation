# NLP 2 Project 2

This project implements a seq2seq model with attention for neural machine translation.
The model class provides either a GRU or a (non-recurrent) positional encoder and uses an LSTM with (bi)linear attention as decoder.
Decoding is done using either greedy search (picking the top scored word as next in the sequence at each decoding step) or using beam search (maintaining a number of sequences and picking the overall most likely one at the end; this slightly improves results while being computationally more expensive).

## Usage

Run `./setup.sh` to 
1. download the training data and dependencies
2. preprocess training data into bytepair encodings [(Sennrich et. al., 2016)](http://www.aclweb.org/anthology/P16-1162)

Run `train.py` to train the seq2seq model
