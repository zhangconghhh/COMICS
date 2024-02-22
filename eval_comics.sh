#! /bin/bash

PYTHON="anaconda3/envs/adet1/bin/python"

OMP_NUM_THREADS=1 $PYTHON -W ignore tools/test.py \
    --config-file training_dir/blenmask_comics/config.yaml \
    --num-gpus 1 \
    OUTPUT_DIR  test_dir/blenmask_comics

