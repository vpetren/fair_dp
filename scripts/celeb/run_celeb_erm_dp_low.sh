#!/bin/bash

SEED=2
OUT=logs$SEED/logs_celeb_erm_dp_low
EPS=10

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset celebA --root_dir data --algorithm ERM --log_dir $OUT --log_every 500 --differentially_private --dp_max_grad_norm 1.2 --target_epsilon $EPS --batch_size 64 > $OUT.out

