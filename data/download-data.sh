#!/bin/bash

wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/test.zip
wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/train.zip
wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/val.zip

unzip test.zip
unzip train.zip
unzip val.zip

rm test.zip
rm val.zip
rm train.zip

