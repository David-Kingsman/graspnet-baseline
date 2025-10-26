#!/bin/bash

# GraspNet 训练命令
# 数据集路径: /home/zekaijin/graspnet-baseline/graspnet_training_datasets

DATASET_ROOT="/home/zekaijin/graspnet-baseline/graspnet_training_datasets"
CAMERA="realsense"  # 可以选择 realsense 或 kinect
LOG_DIR="./logs"

echo "🚀 开始训练 GraspNet 模型"
echo "================================"
echo "📊 训练配置:"
echo "   - 数据集路径: $DATASET_ROOT"
echo "   - 相机类型: $CAMERA"
echo "   - 日志目录: $LOG_DIR"
echo "================================"
echo ""

# 创建日志目录
mkdir -p "$LOG_DIR"

# 开始训练
python train.py \
    --dataset_root "$DATASET_ROOT" \
    --camera "$CAMERA" \
    --log_dir "$LOG_DIR" \
    --max_epoch 18 \
    --batch_size 2 \
    --learning_rate 0.001 \
    --num_point 20000 \
    --num_view 300

echo ""
echo "🎉 训练完成!"
echo "📁 模型保存在: $LOG_DIR/checkpoint.tar"
echo "📊 查看训练日志: $LOG_DIR/log_train.txt"
echo "📈 启动TensorBoard: tensorboard --logdir $LOG_DIR"
