#!/bin/bash

# GraspNet 训练启动脚本

echo "🚀 GraspNet 训练启动脚本"
echo "================================"

# 检查参数
if [ $# -lt 2 ]; then
    echo "❌ 使用方法: $0 <dataset_root> <camera> [log_dir]"
    echo "   参数说明:"
    echo "     dataset_root: 数据集根目录路径"
    echo "     camera: 相机类型 (realsense/kinect)"
    echo "     log_dir: 日志目录 (可选，默认: ./logs)"
    echo ""
    echo "   示例:"
    echo "     $0 /path/to/graspnet realsense"
    echo "     $0 /path/to/graspnet kinect ./my_logs"
    exit 1
fi

DATASET_ROOT=$1
CAMERA=$2
LOG_DIR=${3:-"./logs"}

echo "📊 训练配置:"
echo "   - 数据集路径: $DATASET_ROOT"
echo "   - 相机类型: $CAMERA"
echo "   - 日志目录: $LOG_DIR"
echo ""

# 检查数据集路径
if [ ! -d "$DATASET_ROOT" ]; then
    echo "❌ 错误: 数据集路径不存在: $DATASET_ROOT"
    exit 1
fi

# 检查相机类型
if [ "$CAMERA" != "realsense" ] && [ "$CAMERA" != "kinect" ]; then
    echo "❌ 错误: 相机类型必须是 'realsense' 或 'kinect'"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查依赖
echo "🔍 检查依赖包..."
python check_requirements.py

# 开始训练
echo ""
echo "🎯 开始训练..."
echo "================================"

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
