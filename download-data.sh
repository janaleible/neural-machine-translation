#!/bin/bash

if [ ! -d "data" ]; then

    mkdir data/

    wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/test.zip
    wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/train.zip
    wget https://uva-slpl.github.io/nlp2/resources/project_nmt/data/val.zip

    unzip test.zip -d data/
    unzip train.zip -d data/
    unzip val.zip -d data/

    rm test.zip
    rm val.zip
    rm train.zip

fi
