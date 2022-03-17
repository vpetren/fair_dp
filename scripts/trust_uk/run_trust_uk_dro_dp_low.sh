#!/bin/bash

MODEL=distilbert-base-uncased
SEED=2
OUT=logs$SEED/logs_trust_uk_dro_dp_low
LANG=uk
EPS=10
ALG=groupDRO

python -W ignore::UserWarning run_expt.py --seed $SEED --freeze_bert --dataset trustpilot --split_scheme $LANG --root_dir data --algorithm $ALG --model $MODEL --batch_size 8  --virtual_batch_size 16 --log_every 500 --differentially_private --dp_max_grad_norm 1.2 --target_epsilon $EPS --log_dir $OUT > $OUT.out

