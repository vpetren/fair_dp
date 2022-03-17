#!/bin/bash

MODEL=distilbert-base-uncased
SEED=2
OUT=logs$SEED/logs_blog_dro_dp_low
EPS=10

python -W ignore::UserWarning run_expt.py --seed $SEED --dataset blog --root_dir data --algorithm groupDRO --model $MODEL --batch_size 8 --virtual_batch_size 16 --differentially_private --dp_max_grad_norm 1.2 --target_epsilon $EPS --freeze_bert --log_every 500 --log_dir $OUT > $OUT.out

