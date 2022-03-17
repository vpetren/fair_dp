#!/bin/bash

SEED=2
OUT=logs$SEED/logs_celeb_dro_dp_mid
EPS=5

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset celebA --root_dir data --algorithm groupDRO --log_dir $OUT --log_every 500 --differentially_private --dp_max_grad_norm 1.2 --target_epsilon $EPS --batch_size 64 > $OUT.out

