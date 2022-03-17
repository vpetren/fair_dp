#!/bin/bash

SEED=2
OUT=logs$SEED/logs_celeb_dro

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset celebA --root_dir data --algorithm groupDRO --log_dir $OUT --n_epochs 20 --log_every 500 --batch_size 64 > $OUT.out

