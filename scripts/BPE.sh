#!/usr/bin/env bash

if [ ! -d "lib" ]; then
    mkdir lib
fi
if [ ! -d "lib/subword-nmt-master" ]; then
    wget https://github.com/rsennrich/subword-nmt/archive/master.zip
    unzip master.zip -d lib/
    rm master.zip
fi

lib/subword-nmt-master/learn_joint_bpe_and_vocab.py --input data/train.en data/train.fr -s 10000 -o data/bpe --write-vocabulary data/vocab.en data/vocab.fr

lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.en --vocabulary-threshold 50 < data/train.token.en > data/BPE/train/train.BPE.en
lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.fr --vocabulary-threshold 50 < data/train.token.fr > data/BPE/train/train.BPE.fr

python2 lib/nematus/build_dictionary.py data/BPE/train/train.BPE.en data/BPE/train/train.BPE.fr

lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.en --vocabulary-threshold 50 < data/val.token.en > data/BPE/valid/val.BPE.en
lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.fr --vocabulary-threshold 50 < data/val.token.fr > data/BPE/valid/val.BPE.fr

lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.en --vocabulary-threshold 50 < data/test.token.en > data/BPE/test/test.BPE.en
lib/subword-nmt-master/apply_bpe.py -c data/bpe --vocabulary data/vocab.fr --vocabulary-threshold 50 < data/test.token.fr > data/BPE/test/test.BPE.fr
