#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT-py/`

DATA_PATH_0=`realpath ${SCRIPT_PATH}/folds_эвенкийский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_0}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_0}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_0}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_0}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_0}/${FOLD}/
done

DATA_PATH_1=`realpath ${SCRIPT_PATH}/folds_селькупский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_1}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_1}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_1}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_1}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_1}/${FOLD}/
done

DATA_PATH_2=`realpath ${SCRIPT_PATH}/folds_вепсский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_2}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_2}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_2}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_2}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_2}/${FOLD}/
done

DATA_PATH_3=`realpath ${SCRIPT_PATH}/folds_карельский_кар/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_3}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_3}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_3}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_3}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_3}/${FOLD}/
done

DATA_PATH_4=`realpath ${SCRIPT_PATH}/folds_карельский_ливвик/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_4}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_4}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_4}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_4}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_4}/${FOLD}/
done

DATA_PATH_5=`realpath ${SCRIPT_PATH}/folds_карельский_людик/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/preprocess.py -train_src ${DATA_PATH_5}/${FOLD}/src_token_chars_pos.train \
    -train_tgt ${DATA_PATH_5}/${FOLD}/trg_token_chars_pos.train -valid_src ${DATA_PATH_5}/${FOLD}/src_token_chars_pos.valid \
     -valid_tgt ${DATA_PATH_5}/${FOLD}/trg_token_chars_pos.valid -save_data ${DATA_PATH_5}/${FOLD}/
done
