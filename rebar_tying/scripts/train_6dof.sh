#!/bin/bash

# 6DoF Pose Estimation Training Script

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
DATASET_ROOT="$PROJECT_ROOT/rebar_tying/datasets"
LOG_DIR="$PROJECT_ROOT/rebar_tying/runs/6dof_pose_training"
CAMERA="realsense"
EPOCHS=50
BATCH_SIZE=4
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
NUM_POINTS=20000
NUM_WORKERS=4

echo "🚀 Starting 6DoF Pose Estimation Training"
echo "Dataset: $DATASET_ROOT | Log: $LOG_DIR | Epochs: $EPOCHS"
echo ""

# Pre-flight checks
if [ ! -d "$DATASET_ROOT/scenes" ]; then
    echo "❌ Error: Dataset not found: $DATASET_ROOT/scenes"
    exit 1
fi

# Run training (cd to project root first)
cd "$PROJECT_ROOT"

# Use python from current environment (assumes conda environment is activated)
python3 rebar_tying/scripts/train_6dof_pose.py \
    --dataset_root $DATASET_ROOT \
    --camera $CAMERA \
    --log_dir $LOG_DIR \
    --max_epoch $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --num_point $NUM_POINTS \
    --num_workers $NUM_WORKERS \
    --w_rot 1.0 \
    --w_trans 10.0 \
    --use_symmetry_loss \
    --lr_decay_steps "20,35,45" \
    --lr_decay_rates "0.1,0.1,0.1"

echo ""
echo "✅ Training completed! Check: $LOG_DIR"
