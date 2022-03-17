#!/bin/bash

SEED=2
OUT=logs$SEED/logs_celeb_erm

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset celebA --root_dir data --algorithm ERM --log_dir $OUT --n_epochs 20 --log_every 100 --batch_size 64 > $OUT.out

