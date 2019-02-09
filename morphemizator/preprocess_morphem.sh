#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
DATA_PATH=`realpath ${SCRIPT_PATH}/morphemizator/folds/`

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${SCRIPT_PATH}/preprocess.py -train_src ${DATA_PATH}/${FOLD}/src_tokens_chars.train \
    -train_tgt ${DATA_PATH}/${FOLD}/trg_tokens_chars.train -valid_src ${DATA_PATH}/${FOLD}/src_tokens_chars.valid \
     -valid_tgt ${DATA_PATH}/${FOLD}/trg_tokens_chars.valid -save_data ${DATA_PATH}/${FOLD}/
done