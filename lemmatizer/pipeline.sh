#!/usr/bin/env bash
set -e

./train.sh
./predict.sh
./estimate.sh