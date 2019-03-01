#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language эвенкийский --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language селькупский --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language вепсский --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_кар --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_ливвик --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_людик --bert_path /home/m.domrachev/Models/multi_cased_L-12_H-768_A-12