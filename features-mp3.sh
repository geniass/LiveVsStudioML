#!/bin/bash

yaafe -r 44100 -c config.txt -b mfcc live/*.mp3 studio/*.mp3 | tee yaafe.log
