#!/bin/bash

echo "🚀 start graspnet backbone training"
echo "=================================="

python3 rebar_tying/scripts/train_graspnet_backbone.py \
    --data_dir rebar_tying/datasets/scenes \
    --epoch 100 \
    --batch_size 4 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --output_dir rebar_tying/runs/graspnet_backbone_v1 \
    --max_points 20000 \
    --augment \
    --bn_decay_step 2 \
    --bn_decay_rate 0.5 \
    --lr_decay_steps "30,60,80" \
    --lr_decay_rates "0.1,0.1,0.1" \
    --num_workers 4

echo "✅ train graspnet backbone completed!"
echo "📊 TensorBoard logs are saved in: rebar_tying/runs/graspnet_backbone_v1/"
echo "💾 The best model is saved in: rebar_tying/runs/graspnet_backbone_v1/best_model.pth"
