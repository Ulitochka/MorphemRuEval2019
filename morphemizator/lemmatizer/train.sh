#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT-py/`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds_селькупский/`

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${COMMAND_PATH}/train.py -data ${DATA_PATH}/${FOLD}/ -save_model ${DATA_PATH}/${FOLD}/ \
    -global_attention mlp -encoder_type brnn \
    -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -gpu_ranks 0 \
    -optim adam -learning_rate 0.001 -train_steps 8000 -valid_steps 2000 -batch_size=512
done
