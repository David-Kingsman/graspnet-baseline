#!/bin/bash

DATASET_ROOT="/home/zekaijin/graspnet-baseline/graspnet_training_datasets"
CAMERA="realsense"
LOG_DIR="./logs"

echo "🚀 GraspNet optimized training configuration"
echo "================================"
echo "⚡ optimization settings:"
echo "   - batch size: 8 (balance speed and stability)"
echo "   - data loading: 8 workers (avoid too many processes)"
echo "   - point cloud number: 20000"
echo "================================"
echo ""

# create log directory
mkdir -p "$LOG_DIR"

# start training
python train.py \
    --dataset_root "$DATASET_ROOT" \
    --camera "$CAMERA" \
    --log_dir "$LOG_DIR" \
    --max_epoch 18 \
    --batch_size 8 \
    --learning_rate 0.001 \
    --num_point 20000 \
    --num_view 300 \
    --num_workers 8

echo ""
echo "🎉 training completed!"
echo "📁 model saved in: $LOG_DIR/checkpoint.tar"
echo "📊 view training log: $LOG_DIR/log_train.txt"
echo "📈 start TensorBoard: tensorboard --logdir $LOG_DIR"
