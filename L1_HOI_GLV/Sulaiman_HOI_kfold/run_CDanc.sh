#!/bin/bash

# unpack NSM codes
tar -xzf glove_L1.tar.gz
# unpack data 
tar -xzf folds_CDanc.tar.gz

# run your script
python3 CDanc_20fold.py $1

