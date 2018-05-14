./learn_joint_bpe_and_vocab.py --input train.en train.fr -s 10000 -o bpe --write-vocabulary vocab.en vocab.fr

./apply_bpe.py -c bpe --vocabulary vocab.en --vocabulary-threshold 50 < train.en > train.BPE.en
./apply_bpe.py -c bpe --vocabulary vocab.fr --vocabulary-threshold 50 < train.fr > train.BPE.fr

python2 build_dictionary.py train.BPE.en train.BPE.fr

./apply_bpe.py -c bpe --vocabulary vocab.en --vocabulary-threshold 50 < val.en > val.BPE.en
./apply_bpe.py -c bpe --vocabulary vocab.fr --vocabulary-threshold 50 < val.fr > val.BPE.fr

./apply_bpe.py -c bpe --vocabulary vocab.en --vocabulary-threshold 50 < test.en > test.BPE.en
./apply_bpe.py -c bpe --vocabulary vocab.fr --vocabulary-threshold 50 < test.fr > test.BPE.fr

