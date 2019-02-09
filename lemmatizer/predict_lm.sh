#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds/`



for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${SCRIPT_PATH}/lemmatizer.py --model ${DATA_PATH}/${FOLD}/_step_15000.pt --src ${DATA_PATH}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done
