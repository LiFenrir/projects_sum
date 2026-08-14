#!/bin/bash


# * usage: ./train.sh CONFIG_NAME NGPUS_PER_NODE

config_name=${1}
ngpus_per_node=${2}
PY_ARGS=${@:3}


# cd to the directory of the script
cd $(dirname $(realpath $0))

export WANDB_MODE=offline
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"

if [[ "$PY_ARGS" == *"--resume"* ]]; then
  echo "Resuming training..."
  torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name $PY_ARGS
else
  echo "Overwriting training..."
  torchrun --standalone --nnodes=1 --nproc_per_node=$ngpus_per_node scripts/train_pytorch.py $config_name --exp_name $config_name --overwrite $PY_ARGS
fi
