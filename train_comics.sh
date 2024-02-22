#! /bin/bash

PYTHON="/anaconda3/envs/COMICS/bin/python"


OMP_NUM_THREADS=3 CUDA_LAUNCH_BLOCKING=1 $PYTHON -W ignore tools/train.py \
    --config-file configs/BlendMask/R_50_1x.yaml \
    --num-gpus 1 \
    OUTPUT_DIR training_dir/blenmask_comics
