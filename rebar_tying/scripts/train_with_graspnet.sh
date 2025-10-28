#!/bin/bash

python3 rebar_tying/scripts/train_graspnet_backbone.py \
    --data_dir rebar_tying/datasets/scenes \
    --epoch 50 \
    --batch_size 4 \
    --lr 0.001 \
    --output_dir rebar_tying/runs/graspnet_backbone \
    --max_points 20000 \
    --augment

