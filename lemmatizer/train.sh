#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT-py/`

DATA_PATH_0=`realpath ${SCRIPT_PATH}/folds_эвенкийский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_0}/${FOLD}/ -save_model ${DATA_PATH_0}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done

DATA_PATH_1=`realpath ${SCRIPT_PATH}/folds_селькупский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_1}/${FOLD}/ -save_model ${DATA_PATH_1}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done

DATA_PATH_2=`realpath ${SCRIPT_PATH}/folds_вепсский/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_2}/${FOLD}/ -save_model ${DATA_PATH_2}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done

DATA_PATH_3=`realpath ${SCRIPT_PATH}/folds_карельский_кар/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_3}/${FOLD}/ -save_model ${DATA_PATH_3}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done

DATA_PATH_4=`realpath ${SCRIPT_PATH}/folds_карельский_ливвик/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_4}/${FOLD}/ -save_model ${DATA_PATH_4}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done

DATA_PATH_5=`realpath ${SCRIPT_PATH}/folds_карельский_людик/`
for FOLD in 0 1 2 3 4
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH_5}/${FOLD}/ -save_model ${DATA_PATH_5}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done
