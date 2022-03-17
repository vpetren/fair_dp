#!/bin/bash

MODEL=distilbert-base-uncased
SEED=2
OUT=logs$SEED/logs_blog_dro

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset blog --root_dir data --algorithm groupDRO --model $MODEL --freeze_bert --batch_size 16 --log_every 500 --log_dir $OUT > $OUT.out

