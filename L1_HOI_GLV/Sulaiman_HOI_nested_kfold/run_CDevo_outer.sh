#!/bin/bash

# unpack NSM codes
tar -xzf glove_L1.tar.gz
# unpack data 
tar -xzf folds_CDevo.tar.gz

# run your script
python3 CDevo_20fold_outer.py $1

