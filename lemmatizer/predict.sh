#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

DATA_PATH_0=`realpath ${SCRIPT_PATH}/folds_эвенкийский/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_0}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_0}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_0}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done


DATA_PATH_1=`realpath ${SCRIPT_PATH}/folds_селькупский/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_1}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_1}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_1}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done


DATA_PATH_2=`realpath ${SCRIPT_PATH}/folds_вепсский/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_2}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_2}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_2}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done


DATA_PATH_3=`realpath ${SCRIPT_PATH}/folds_карельский_кар/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_3}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_3}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_3}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done


DATA_PATH_4=`realpath ${SCRIPT_PATH}/folds_карельский_ливвик/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_4}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_4}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_4}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done


DATA_PATH_5=`realpath ${SCRIPT_PATH}/folds_карельский_людик/`
for FOLD in 0 1 2 3 4
do
    PYTHONPATH=../ python3 ${SCRIPT_PATH}/lemmatizer.py --beam_size 1 --model ${DATA_PATH_5}/${FOLD}/_step_8000.pt \
    --src ${DATA_PATH_5}/${FOLD}/EXERCISE_INPUT.txt --output ${DATA_PATH_5}/${FOLD}/EXERCISE_PRED.txt --attn_debug
done
