#!/usr/bin/env bash

if [ ! -d "lib" ]; then
    mkdir lib
fi
if [ ! -d "lib/moses" ]; then
    wget -P lib/moses https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/python-tokenizer/moses.py
fi

if [ ! -d "lib/nematus" ]; then
    wget -P lib/nematus https://raw.githubusercontent.com/EdinburghNLP/nematus/master/data/build_dictionary.py
fi

wget -P lib/ https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl