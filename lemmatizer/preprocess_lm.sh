#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT_py/`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds/`

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH}/${FOLD}/
done