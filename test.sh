#!/bin/bash -i

source ./env

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_dir $DATA_DIR \
  --model_dir_root $MODEL_ROOT_DIR \
  --pretrain_path=./gpt2	\
  "$@"
