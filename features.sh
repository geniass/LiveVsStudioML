#!/bin/bash

#for f in studio/*; do echo "$f"; mpg123 -w wav/${f//[[:blank:]]/}.wav "$f"; done

for f in wav/live_test/*
do
    echo $f >> yaafe-files.txt
done

for f in wav/studio/*
do
    echo $f >> yaafe-files.txt
done

yaafe -r 44100 -c config.txt -i yaafe-files.txt -b mfcc | tee yaafe.log
