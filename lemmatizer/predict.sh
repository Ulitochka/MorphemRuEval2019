#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds/`



for FOLD in 0
do
    python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH}/${FOLD}/_step_100.pt --src ${DATA_PATH}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done
