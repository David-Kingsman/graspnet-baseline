# Rebar Node Tying System

A 6DoF pose estimation system for automated rebar tying operations, built on GraspNet backbone with point cloud processing.

## Overview

This system estimates 6DoF poses of rebar crossing nodes from point clouds, enabling robotic tying operations. It uses YOLO for node detection and GraspNet's Pointnet2Backbone for hierarchical feature extraction.

**Key Features:**
- Point cloud-based 6DoF pose estimation (SE(3))
- YOLO node detection with bounding box extraction
- GraspNet backbone for robust feature learning
- End-to-end pose regression from local point clouds

---

## Current Status

### ✅ Completed

1. **Data Preparation** - YOLO-based local point cloud extraction with automatic pose estimation via cylinder fitting, generating GraspNet-format training data (RGB+Depth+mask+pose+pointclouds)

2. **Training Pipeline** - Full training system with GraspNet backbone and 6DoF pose regression (point cloud input → SE(3) pose output)

3. **Advanced Features** - Symmetry-aware loss, SE(3) projection, learning rate scheduling, TensorBoard monitoring, checkpoint recovery, ETA tracking

4. **Inference & Evaluation** - Batch evaluation with CSV export, visualization support, and comprehensive error metrics

---

## Project Structure

```
rebar_tying/
├── scripts/
│   ├── process_training_data.py      # Data processing (RGB+Depth→point cloud+pose)
│   ├── train_6dof_pose.py            # Training script with GraspNet backbone
│   ├── inference_pose.py             # Inference & evaluation tool
│   ├── cylinder_fitting.py           # Cylinder fitting for pose estimation
│   ├── check_pose_consistency.py      # Pose consistency checker
│   └── train_6dof.sh                 # Training launcher
├── datasets/scenes/  # Dataset (organized by scene)
│   ├── scene_vertical/               # Vertical orientation scenes
│   ├── scene_incline/                # Incline orientation scenes
│   └── [other scenes]/               # Additional scene variations
├── texture_suppression_model/        # YOLO model for node detection
└── runs/6dof_pose_training/         # Training results
```

---

## Dataset Statistics

**Organization**: Scenes grouped by orientation (Vertical/Incline)  
**Data Format**: GraspNet-compatible structure  
**Point Clouds**: Local point clouds extracted from YOLO bounding boxes  
**Pose Estimation**: Cylinder fitting from two intersecting rebars  
**Storage**: Compressed point clouds (.npy) + visualization format (.ply)

---

## System Architecture

```
RGB Image + Depth Image
    ↓
YOLO Node Detection
    ↓
Bounding Box Extraction
    ↓
Depth → Point Cloud Conversion
    ↓
Local Point Cloud (Input)
    ↓
GraspNet Backbone (Pointnet2Backbone)
    ↓ 4×SA (Set Abstraction) + 2×FP (Feature Propagation)
Pose Regression Head
    ↓
6DoF Pose (4×4 SE(3)) (Output)
```

**Technical Highlights:**
- Hierarchical feature learning with 4 SA layers (2048→1024→512→256 points)
- 2 FP layers for feature propagation
- **Symmetry-aware rotation loss**: Handles 180° rotations and axis flips for rebar cross-structure symmetry
- **SE(3) projection**: SVD-based orthogonalization ensures valid rotation matrices
- **Adaptive loss weighting**: Configurable rotation/translation loss balance (default: rot=5.0, trans=1.0)
- **Learning rate scheduling**: Multi-stage decay (default: 20, 35, 45 epochs)
- **BN momentum scheduling**: Decay from 0.5 to 0.001 for stable training

---

## Key Components

### 1. Data Processing: `process_training_data.py`

Converts RGB+Depth images to GraspNet-format training data with point clouds and poses.

**Pipeline**: YOLO bbox detection → Depth-to-point-cloud conversion → PCA-based pose estimation → GraspNet format output

**Usage:**
```bash
python3 rebar_tying/scripts/process_training_data.py \
    --depth_dir rebar_tying/texture_suppression_model/images/rebar_joint_pose_estimation/Nano0711 \
    --output_dir rebar_tying/datasets \
    --yolo_model rebar_tying/texture_suppression_model/runs/pose/train2/weights/best.pt \
    --conf 0.35 \
    --imgsz 960 \
    --device cuda:0 \
    --min_points 200
```

**Output Structure:**
- Scenes organized by orientation (`scene_vertical/`, `scene_incline/`)
- Each scene contains: `rgb/`, `depth/`, `label/`, `meta/`, `pointclouds/`, `labels_xyxy_depth/`
- Point clouds saved as both `.npy` (training) and `.ply` (visualization)
- Camera intrinsics saved per scene (`camK.npy`)
- YOLO detection labels saved for reproducibility (`labels_xyxy_depth/*.txt`)

**Data Format:**
- Point clouds: Local point clouds extracted from YOLO bounding boxes
- Poses: 4×4 SE(3) transformation matrices (estimated via cylinder fitting)
- Supports both Vertical and Incline orientations (automatic subdirectory detection)

---

### 2. Training: `train_6dof_pose.py`

End-to-end pose estimation training with enhanced GraspNet backbone.

**Network Architecture:**
- Input: Point Cloud (N, 3) → Downsampled to 20K points (configurable)
- Backbone: GraspNet Pointnet2Backbone (hierarchical feature extraction)
- Global Pooling: Max + Average pooling → (B, 2C) features
- Pose Head: 512→256→128→16 → Reshape to (B, 4, 4) pose matrix
- **SE(3) Projection**: SVD-based orthogonalization ensures valid rotations

**Loss Function:**
```python
Loss = w_trans × Translation_Loss + w_rot × Rotation_Loss
```
- Translation Loss: Smooth L1 loss (default weight: 1.0)
- Rotation Loss: SO(3) geodesic distance in radians (default weight: 5.0)
- **Symmetry-aware mode**: Minimizes rotation error over symmetric transformations (optional, `--use_symmetry_loss`)

**Advanced Training Features:**
- Learning rate scheduling (default: decay at epochs 20, 35, 45)
- BN momentum scheduling (0.5→0.001)
- TensorBoard real-time monitoring
- Checkpoint recovery system (resume from `checkpoint.tar`)
- Multi-worker data loading
- Gradient clipping (max_norm=1.0)
- Training progress tracking with ETA estimation

**Usage:**
```bash
# Quick start with default settings
bash rebar_tying/scripts/train_6dof.sh

# Manual training with custom parameters
python3 rebar_tying/scripts/train_6dof_pose.py \
    --dataset_root rebar_tying/datasets \
    --camera realsense \
    --log_dir rebar_tying/runs/6dof_pose_training \
    --max_epoch 50 \
    --batch_size 4 \
    --learning_rate 0.0001 \
    --weight_decay 0.0001 \
    --num_point 20000 \
    --w_rot 5.0 \
    --w_trans 1.0 \
    --use_symmetry_loss \
    --lr_decay_steps "20,35,45" \
    --lr_decay_rates "0.1,0.1,0.1"

# Monitor with TensorBoard
tensorboard --logdir rebar_tying/runs/6dof_pose_training
```

---

## Training Results

### ✅ **Training Pipeline Completed**

**Status**: Training script fully implemented with advanced features

**Key Features:**
- ✅ **SE(3) Projection**: SVD-based orthogonalization ensures valid rotation matrices
- ✅ **Symmetry-aware Loss**: Handles rebar cross-structure symmetry (180° rotations, axis flips)
- ✅ **Learning Rate Scheduling**: Multi-stage decay (default: 20, 35, 45 epochs)
- ✅ **BN Momentum Scheduling**: From 0.5 to 0.001 decay
- ✅ **TensorBoard Monitoring**: Real-time training visualization
- ✅ **Checkpoint Recovery**: Resume training from interruptions (`checkpoint.tar`)
- ✅ **Data Preprocessing Consistency**: Unified training/inference preprocessing
- ✅ **Progress Tracking**: ETA estimation and detailed epoch summaries

**Network Architecture:**
```
Point Cloud (N, 3) 
  → GraspNet Backbone (4×SA: 2048→1024→512→256 + 2×FP) 
  → Global Pooling (Max + Avg) → (B, 2C)
  → Pose Head (512→256→128→16)
  → Reshape to (B, 4, 4)
  → SVD-based SE(3) Projection
  → 6DoF Pose (4×4 SE(3)) - GUARANTEED ORTHOGONAL
```

**Loss Function:**
```python
Loss = w_trans × SmoothL1(translation) + w_rot × Geodesic(rotation)
# Default: w_trans=1.0, w_rot=5.0
# Optional: --use_symmetry_loss for symmetry-aware rotation error
```

---

## Quick Start

### 1. Data Generation
```bash
python3 rebar_tying/scripts/process_training_data.py \
    --depth_dir <depth_dir_with_Vertical_and_Incline_subdirs> \
    --output_dir rebar_tying/datasets \
    --yolo_model rebar_tying/texture_suppression_model/runs/pose/train2/weights/best.pt \
    --conf 0.35 \
    --imgsz 960 \
    --device cuda:0 \
    --min_points 200
```

### 2. Training
```bash
# Quick start with default configuration
bash rebar_tying/scripts/train_6dof.sh

# Or customize parameters directly
python3 rebar_tying/scripts/train_6dof_pose.py \
    --dataset_root rebar_tying/datasets \
    --log_dir rebar_tying/runs/6dof_pose_training \
    --max_epoch 50 \
    --batch_size 4 \
    --use_symmetry_loss
```

### 3. Inference & Evaluation
```bash
# Single point cloud prediction
python3 rebar_tying/scripts/inference_pose.py \
    --model_path rebar_tying/runs/6dof_pose_training/best_model.tar \
    --pointcloud_path rebar_tying/datasets/scenes/scene_vertical/pointclouds/0000_obj0.npy

# Batch evaluation on test set
python3 rebar_tying/scripts/inference_pose.py \
    --model_path rebar_tying/runs/6dof_pose_training/best_model.tar \
    --data_dir rebar_tying/datasets/scenes \
    --num_samples 20 \
    --save_csv evaluation_results.csv

# With visualization (requires Open3D)
python3 rebar_tying/scripts/inference_pose.py \
    --model_path rebar_tying/runs/6dof_pose_training/best_model.tar \
    --data_dir rebar_tying/datasets/scenes \
    --num_samples 10 \
    --viz
```

---

## Technical Specifications

### Data Format

**Point Cloud**: `.npy` format, shape (N, 3), float64, units in meters  
**Pose**: `.mat` format, 4×4 SE(3) transformation matrix (cls_indexes, poses, intrinsic_matrix, factor_depth)

### Camera Intrinsics
- fx = 1734.76, fy = 1734.59
- cx = 632.24, cy = 504.99

### Dataset Organization
- **Scene Structure**: Organized by orientation (Vertical/Incline subdirectories)
- **Data Generation**: Automatic YOLO detection → point cloud extraction → pose estimation
- **Pose Estimation**: Cylinder fitting-based 6DoF pose from two intersecting rebars
- **Format**: GraspNet-compatible (RGB, depth, label, meta, pointclouds)

### Training/Test Split
- **Split Method**: Scene-level 80/20 split (fixed random seed for reproducibility)
- **Validation**: Automatic test set evaluation during training

---

## Future Work

### ✅ Recently Completed
- ✅ Symmetry-aware rotation loss for rebar cross-structure symmetry
- ✅ SE(3) projection for guaranteed orthogonal rotation matrices
- ✅ Advanced training features (LR scheduling, TensorBoard, checkpoint recovery, ETA tracking)
- ✅ Improved inference script with batch evaluation and CSV export
- ✅ Data preprocessing pipeline supporting Vertical/Incline orientations
- ✅ Cylinder fitting-based pose estimation from point clouds

### In Progress
- Performance evaluation and benchmarking on test set
- Hyperparameter tuning (loss weights, learning rates)
- Ablation studies on symmetry-aware loss

### Planned
- Binding tool pose annotation for end-to-end tying control
- Real-time inference optimization
- ROS integration for robotic arm control
- Multi-scene data augmentation strategies

---

## Performance Targets

- **Translation Error**: < 0.01m
- **Rotation Error**: < 5°
- **Training Time**: < 2 hours (GPU)
- **Real-time Inference**: > 10 Hz

**System Capabilities:**
- ✅ YOLO Detection (automatic node detection with confidence filtering)
- ✅ Point Cloud Processing (local point cloud extraction from bounding boxes)
- ✅ GraspNet Backbone Training (hierarchical feature learning)
- ✅ **SE(3) Pose Estimation** (SVD-based orthogonalization)
- ✅ **Symmetry-aware Loss** (handles rebar cross-structure symmetry)
- ✅ **Advanced Training Features** (LR scheduling, TensorBoard, checkpoint recovery, ETA tracking)
- ✅ Inference & Evaluation Tools (batch evaluation, CSV export, visualization)
- ⏳ ROS Integration (planned)

---

## Research Context

**Goal**: Point cloud-based 6DoF pose estimation for rebar node tying operations  
**Approach**: Bounding box → Local point cloud → GraspNet Backbone → 6DoF Pose Regression  
**Future Extension**: Direct binding tool pose prediction for end-to-end robotic control

**References**:
- GraspNet: Efficient Learning for Robotic Grasping
- PointNet++: Deep Hierarchical Feature Learning on Point Sets
- 6DoF Pose Estimation from Point Clouds

---

**Version**: 8.0  
**Last Updated**: 2025-01-27  
**Status**: Training Pipeline Complete

**Recent Updates:**
- Renamed training script: `train_6dof_pose.py` (replaces `train_graspnet_backbone.py`)
- Added symmetry-aware rotation loss (`--use_symmetry_loss`)
- Improved SE(3) projection with SVD-based orthogonalization
- Enhanced training progress tracking with ETA estimation
- Updated data preprocessing to support Vertical/Incline scene organization
- Added batch evaluation and CSV export to inference script
