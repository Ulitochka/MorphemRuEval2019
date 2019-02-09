#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
DATA_PATH=`realpath ${SCRIPT_PATH}/morphemizator/folds/`

for FOLD in 0
do
    python3 ${SCRIPT_PATH}/train.py -encoder_type brnn -gpu_ranks 0 -global_attention mlp -encoder_type brnn -dropout 0.7 -src_word_vec_size=256 -tgt_word_vec_size=256 -dec_rnn_size=512 -enc_rnn_size=512 -optim adam -learning_rate 0.001 -data ${DATA_PATH}/${FOLD}/ -save_model ${DATA_PATH}/${FOLD}/ -train_steps 25000 -valid_steps 1000
done
