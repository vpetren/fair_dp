#!/bin/bash

MODEL=distilbert-base-uncased
SEED=2
OUT=logs$SEED/logs_trust_uk_dro
LANG=uk
ALG=groupDRO

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset trustpilot --split_scheme $LANG --root_dir data --algorithm $ALG --model $MODEL --batch_size 16 --log_every 500 --freeze_bert --log_dir $OUT > $OUT.out

