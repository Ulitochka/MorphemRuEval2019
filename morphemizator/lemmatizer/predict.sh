#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds_селькупский/`



for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH}/${FOLD}/_step_8000.pt --src ${DATA_PATH}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done