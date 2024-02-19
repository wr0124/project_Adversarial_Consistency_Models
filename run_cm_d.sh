#!/bin/bash

iter_size="8"
set -x

for it in $iter_size
do
    python3 cm_d.py \
    --data_dir "/data3/juliew/4fin/dataset/trainB/" \
    --test_data_dir "/data3/juliew/4fin/dataset/testB/"  \
    --image_size 64 64  \
    --batch_size 8  \
    --num_workers 6 \
    --max_steps 200_000 \
    --sample_every_n_epochs 2 \
    --devices 1 \
    --device_cuda "cuda:1" \
    --lr  1e-4 \
    --iter_size "${it}" \
    --num_samples 6 \
    --env "cm_d0.9_1unet_bs8*${it}_test_lr4"
done

