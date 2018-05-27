#!/usr/bin/env bash

if [ ! -d "lib" ]; then
    mkdir lib
fi
if [ ! -d "lib/meteor" ]; then
    wget -P lib/meteor/ http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz

    tar -xzf lib/meteor/meteor-1.5.tar.gz -C lib/meteor/
    rm lib/meteor/meteor-1.5.tar.gz
fi

java -jar -Xmx2G lib/meteor/meteor-1.5/meteor-1.5.jar output/test-hyp.txt output/test-ref.txt
