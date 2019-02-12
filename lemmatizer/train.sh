#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT-py/`
DATA_PATH=`realpath ${SCRIPT_PATH}/folds/`

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${COMMAND_PATH}/train.py -global_attention mlp -encoder_type brnn -learning_rate_decay 0.1 -start_decay_steps 6000 -decay_steps 4000 -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -optim adam -learning_rate 0.001 -data ${DATA_PATH}/${FOLD}/ -save_model ${DATA_PATH}/${FOLD}/ -train_steps 150 -save_checkpoint_steps 100
done
