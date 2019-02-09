#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/OpenNMT_py/`
DATA_PATH=`realpath ${SCRIPT_PATH}/lemmatizer/folds/`

for FOLD in 0 1 2 3 4 5 6 7 8 9
do
    python3 ${COMMAND_PATH}/train.py -gpu_ranks 0 -global_attention mlp -encoder_type brnn -learning_rate_decay 0.1 -start_decay_steps 6000 -decay_steps 4000 -dropout 0.5 -src_word_vec_size=500 -tgt_word_vec_size=500 -dec_rnn_size=1024 -enc_rnn_size=1024 -optim adam -learning_rate 0.001 -data ${DATA_PATH}/${FOLD}/ -save_model ${DATA_PATH}/${FOLD}/ -train_steps 15000 -save_checkpoint_steps 1000
done
