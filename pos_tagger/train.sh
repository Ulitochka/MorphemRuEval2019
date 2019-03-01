#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`

PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language эвенкийский
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language селькупский
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language вепсский
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_кар
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_ливвик
PYTHONPATH=../ python3 ${SCRIPT_PATH}/bert_sbai_experiment.py --language карельский_людик