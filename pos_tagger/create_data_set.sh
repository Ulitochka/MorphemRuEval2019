#!/usr/bin/env bash
set -e

SCRIPT=$(readlink -f $0)
SCRIPT_PATH=`dirname $SCRIPT`
COMMAND_PATH=`realpath ${SCRIPT_PATH}/`

python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language карельский_людик
python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language карельский_ливвик
python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language карельский_кар
python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language вепсский
python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language селькупский
python3 ${COMMAND_PATH}/pos_data_set_former.py --folds 4 --language эвенкийский
