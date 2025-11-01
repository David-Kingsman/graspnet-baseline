#!/usr/bin/env python3
"""
6DoF Pose Estimation Training Script

"""

import os
import sys
import numpy as np
from datetime import datetime, timedelta
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import cv2
import scipy.io as sio

# Add paths following official style
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from backbone import Pointnet2Backbone
from pytorch_utils import BNMomentumScheduler

# Global configuration for 6DoF pose estimation training
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root directory')
parser.add_argument('--camera', default='realsense', help='Camera type [realsense/kinect]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log_pose', help='Dump dir to save model checkpoint [default: log_pose]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Optimization L2 weight decay [default: 0.0001]')
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='20,35,45', help='When to decay the learning rate (in epochs) [default: 20,35,45]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers [default: 0]')

# Bbox options (当前数据生成已保存局部点云，训练时通常不需要二次裁剪)
parser.add_argument('--use_bbox', action='store_true', default=False, help='Use bounding box for local point cloud (通常不需要，因为数据生成时已保存局部点云)')
parser.add_argument('--bbox_size', type=float, default=0.1, help='Bounding box size in meters [default: 0.1]')

# YOLO bbox options (可选，仅当use_bbox=True时使用，从dataset_root/scenes/scene_*/labels_xyxy_depth/读取)
parser.add_argument('--use_yolo_bbox', action='store_true', default=False, help='Use YOLO 2D bbox to filter points (仅当use_bbox=True时有效，从数据集内labels_xyxy_depth/读取)')
parser.add_argument('--bbox_debug', action='store_true', help='Print verbose logs for bbox/YOLO filtering debug')

# Loss weights (平衡旋转和平移损失)
parser.add_argument('--w_rot', type=float, default=1.0, help='Weight for rotation loss [default: 1.0]')
parser.add_argument('--w_trans', type=float, default=1.0, help='Weight for translation loss [default: 1.0]')
parser.add_argument('--use_symmetry_loss', action='store_true', default=False, help='Use symmetry-aware rotation loss (处理钢筋交叉结构的对称性)')

cfgs = parser.parse_args()

# Global config for 6DoF pose estimation training
EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None else DEFAULT_CHECKPOINT_PATH

# Training time statistics 
TRAINING_START_TIME = None
EPOCH_TIMES = []
BATCH_TIMES = []

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)

LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')
LOG_FOUT.write(str(cfgs)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Worker initialization following official style
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass


class PoseDataset:
    """
    Dataset for 6DoF pose estimation following official GraspNet style
    Supports both global and local (bbox) point clouds
    """
    
    def __init__(self, dataset_root, camera='realsense', split='train', 
                 num_points=20000, use_bbox=False, bbox_size=0.1,
                 use_yolo_bbox=False):
        """
        Args:
            dataset_root: root directory of dataset (包含scenes/scene_*/)
            camera: camera type (兼容参数，当前未使用)
            split: train/test split
            num_points: number of points after sampling
            use_bbox: whether to use bounding box for local point cloud 
                     (通常不需要，因为数据生成时已保存局部点云)
            bbox_size: bounding box size in meters (仅当use_bbox=True时使用)
            use_yolo_bbox: 是否使用YOLO 2D bbox过滤 (仅当use_bbox=True时有效)
                          YOLO标签从 dataset_root/scenes/scene_*/labels_xyxy_depth/读取
        """
        self.dataset_root = dataset_root
        self.camera = camera
        self.split = split
        self.num_points = num_points
        self.use_bbox = use_bbox
        self.bbox_size = bbox_size
        self.use_yolo_bbox = use_yolo_bbox
        
        # Get all scenes - 场景级随机划分（80/20，固定种子保证可复现）
        scenes_dir = os.path.join(dataset_root, 'scenes')
        all_scenes = sorted(glob.glob(os.path.join(scenes_dir, 'scene_*')))
        all_scenes = [s for s in all_scenes if os.path.isdir(s)]
        
        # 固定随机种子 + 场景级 80/20 划分
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_scenes))
        split_idx = int(0.8 * len(all_scenes))
        train_ids, test_ids = perm[:split_idx], perm[split_idx:]
        
        scene_folders = [all_scenes[i] for i in (train_ids if self.split == 'train' else test_ids)]
        print(f"[{self.split}] scenes: {len(scene_folders)} / {len(all_scenes)} total")
        if len(scene_folders) == 0:
            print(f"Warning: No scenes found in {scenes_dir}")
            print(f"Available: {os.listdir(scenes_dir) if os.path.exists(scenes_dir) else 'Directory does not exist'}")
        
        # Build sample list
        self.samples = []
        # Cache camera intrinsics per scene
        self.scene_intrinsics = {}
        for scene_path in scene_folders:
            scene_name = os.path.basename(scene_path)
            # load camera intrinsics if available
            K_path = os.path.join(scene_path, 'camK.npy')
            if os.path.exists(K_path):
                try:
                    self.scene_intrinsics[scene_name] = np.load(K_path).reshape(3,3).astype(np.float32)
                except Exception:
                    pass
            
            # Get point cloud files
            pc_dir = os.path.join(scene_path, 'pointclouds')
            if not os.path.exists(pc_dir):
                continue
                
            pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.npy')))
            
            for pc_file in pc_files:
                basename = os.path.basename(pc_file)
                # Format: XXXX_objY.npy
                parts = basename.replace('.npy', '').split('_')
                frame_idx = int(parts[0])
                obj_idx = int(parts[1].replace('obj', ''))
                
                # Load corresponding meta file
                meta_file = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
                if not os.path.exists(meta_file):
                    continue
                
                try:
                    meta = sio.loadmat(meta_file)
                    poses = meta['poses']  # (N, 4, 4)
                    
                    if obj_idx >= len(poses):
                        continue
                    
                    pose = poses[obj_idx]  # 4x4 transformation matrix
                    
                    self.samples.append({
                        'pc_file': pc_file,
                        'pose': pose,
                        'scene': scene_name,
                        'frame_idx': frame_idx,
                        'obj_idx': obj_idx
                    })
                except Exception as e:
                    print(f"Error loading {meta_file}: {e}")
                    continue
        
        print(f"Total {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load point cloud
        pc = np.load(sample['pc_file']).astype(np.float32)
        
        # Apply bounding box if needed
        if self.use_bbox:
            pc = self._apply_bbox(pc, sample['pose'], sample)
        
        # Sample to fixed size
        pc = self._sample_points(pc, self.num_points)
        
        # Normalize point cloud
        center = np.mean(pc, axis=0)  # get the center point
        pc = pc - center
        
        # Load pose
        pose = sample['pose'].astype(np.float32)
        pose[:3, 3] -= center  # center synchronization
        
        # Convert to torch tensors
        pc = torch.FloatTensor(pc)
        pose = torch.FloatTensor(pose)
        
        return {
            'pointcloud': pc,
            'pose': pose,
            'scene': sample['scene'],
            'frame_idx': sample['frame_idx'],
            'obj_idx': sample['obj_idx']
        }
    
    def _apply_bbox(self, pc, pose, sample=None):
        """Apply bbox filtering.
        Priority: YOLO 2D bbox from dataset (if provided) -> 3D cube around pose center (fallback).
        
        注意：当前数据生成脚本已保存局部点云，通常不需要在训练时再次裁剪。
        此方法仅当use_bbox=True时调用（用于兼容性或特殊情况）。
        """
        # Try YOLO 2D bbox from dataset (数据生成脚本保存的位置)
        if self.use_yolo_bbox and sample is not None:
            scene = sample['scene']
            frame_idx = sample['frame_idx']
            K = self.scene_intrinsics.get(scene, None)
            
            # 从dataset_root/scenes/scene_*/labels_xyxy_depth/####.txt读取
            # 这是process_training_data.py保存的位置（深度分辨率下的像素坐标）
            scene_path = os.path.join(self.dataset_root, 'scenes', scene)
            cand_txts = [
                os.path.join(scene_path, 'labels_xyxy_depth', f"{frame_idx:04d}.txt"),  # 数据生成脚本保存的位置（优先）
                os.path.join(scene_path, f"{frame_idx:04d}.txt"),  # 备用路径
            ]
            
            det_txt = next((p for p in cand_txts if os.path.exists(p)), None)
            if cfgs.bbox_debug:
                print(f"[BBoxDebug] scene={scene} frame={frame_idx:04d} K={'ok' if K is not None else 'None'} label_path={det_txt if det_txt else 'NOT_FOUND'}")
            if K is not None and det_txt is not None:
                bboxes = self._read_yolo_txt_xyxy_pixels(det_txt, scene, frame_idx)
                # 过滤过小/低置信度框，增强稳定性
                bboxes = [(x1,y1,x2,y2,c) for (x1,y1,x2,y2,c) in bboxes
                          if (x2-x1) > 5 and (y2-y1) > 5 and c >= 0.25]
                if cfgs.bbox_debug:
                    print(f"[BBoxDebug] parsed {len(bboxes)} boxes from {os.path.basename(det_txt)}")
                if len(bboxes) > 0:
                    pc2d = self._project_points_to_image(pc, K)
                    # choose best bbox: nearest to projected mean
                    uv_mean = np.mean(pc2d, axis=0)
                    dists = [np.linalg.norm(((bx+ex)/2 - uv_mean[0], (by+ey)/2 - uv_mean[1])) for (bx,by,ex,ey,_) in bboxes]
                    sel = int(np.argmin(dists))
                    x1,y1,x2,y2,_ = bboxes[sel]
                    mask = (pc2d[:,0] >= x1) & (pc2d[:,0] <= x2) & (pc2d[:,1] >= y1) & (pc2d[:,1] <= y2)
                    filtered = pc[mask]
                    if cfgs.bbox_debug:
                        w, h = self._get_rgb_size(scene, frame_idx) or (-1, -1)
                        print(f"[BBoxDebug] img=({w}x{h}) box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) kept_points={len(filtered)} / {len(pc)}")
                    if len(filtered) >= 100:
                        return filtered
        # Fallback: 3D cube around pose center (adaptive)
        # Get object center from pose
        center = pose[:3, 3]
        
        # Start with original bbox size
        current_size = self.bbox_size
        
        # Try different bbox sizes until we get enough points
        for attempt in range(3):  # Try up to 3 different sizes
            bbox_min = center - current_size / 2
            bbox_max = center + current_size / 2
            
            # Filter points within bounding box
            mask = np.all((pc >= bbox_min) & (pc <= bbox_max), axis=1)
            filtered_pc = pc[mask]
            
            # If we have enough points, return
            if len(filtered_pc) >= 100:  # At least 100 points
                return filtered_pc
            
            # Expand bbox size for next attempt
            current_size *= 1.5
        
        # If still no points, return original point cloud (fallback)
        if cfgs.bbox_debug:
            print(f"[BBoxDebug] Fallback: bbox filtering failed, use original point cloud ({len(pc)} pts)")
        else:
            print(f"Warning: Bbox filtering failed, using original point cloud")
        return pc

    def _get_rgb_size(self, scene, frame_idx):
        rgb_path = os.path.join(self.dataset_root, 'scenes', scene, 'rgb', f"{frame_idx:04d}.png")
        img = cv2.imread(rgb_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        return w, h

    def _read_yolo_txt_xyxy_pixels(self, txt_path, scene, frame_idx):
        """读取像素坐标 xyxy（深度分辨率），行格式：class x1 y1 x2 y2 [conf]
        
        根据process_training_data.py，保存的格式是像素xyxy，无需归一化检测
        """
        bboxes = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    # 格式：class x1 y1 x2 y2 [conf]
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    bboxes.append((x1, y1, x2, y2, conf))
        except Exception:
            pass
        return bboxes

    def _project_points_to_image(self, pc, K):
        """Project 3D points (camera coords) to image pixels using intrinsics K.
        pc: (N,3) in meters. Returns (N,2) [u,v].
        """
        X = pc[:,0]; Y = pc[:,1]; Z = np.clip(pc[:,2], 1e-6, None)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return np.stack([u,v], axis=1)

    def _read_yolo_txt(self, txt_path):
        """Read detection txt: lines of 'class x1 y1 x2 y2 [conf]' -> list of tuples.
        Returns [(x1,y1,x2,y2,conf_or_1.0), ...]
        """
        bboxes = []
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # support with or without conf
                        if len(parts) == 5:
                            _, x1, y1, x2, y2 = parts
                            conf = 1.0
                        else:
                            _, x1, y1, x2, y2, conf = parts[:6]
                        bboxes.append((float(x1), float(y1), float(x2), float(y2), float(conf)))
        except Exception:
            pass
        return bboxes
    
    def _sample_points(self, pc, num_points):
        """Sample points to fixed size with robust handling"""
        if len(pc) == 0:
            # Last resort: create minimal dummy points
            print(f"Warning: Empty point cloud detected, creating minimal dummy points")
            # Create a small cube of points instead of all zeros
            dummy_points = np.random.uniform(-0.01, 0.01, (num_points, 3)).astype(np.float32)
            return dummy_points
        elif len(pc) > num_points:
            indices = np.random.choice(len(pc), num_points, replace=False)
            pc = pc[indices]
        elif len(pc) < num_points:
            indices = np.random.choice(len(pc), num_points, replace=True)
            pc = pc[indices]
        return pc
    
    def _normalize_pointcloud(self, pc):
        """Normalize point cloud to zero mean"""
        center = np.mean(pc, axis=0)
        pc = pc - center
        return pc


def collate_fn(batch):
    """ Collate function，已采样定长，无需padding """
    pointclouds = torch.stack([item['pointcloud'] for item in batch])
    poses = torch.stack([item['pose'] for item in batch])
    return {
        'pointcloud': pointclouds,
        'pose': poses
    }


class PoseEstimationNet(nn.Module):
    """
    6DoF pose estimation network using GraspNet backbone
    Following official model design patterns
    """
    
    def __init__(self, input_feature_dim=0):
        super(PoseEstimationNet, self).__init__()
        
        # Use GraspNet backbone
        self.backbone = Pointnet2Backbone(input_feature_dim)
        
        # Pose regression head - 使用 LazyLinear 适配不确定的通道数 2*C
        self.pose_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16)
        )
    
    def forward(self, x, end_points=None):
        """
        Args:
            x: (B, N, 3) point cloud
            end_points: additional end points information
        Returns:
            pose: (B, 4, 4) pose matrix
        """
        # 注意：本工程backbone期望输入(B, N, 3)（参见 models/backbone._break_up_pc），无需转置
        # Extract features through backbone
        seed_features, seed_xyz, end_points = self.backbone(x, end_points)
        
        # Global pooling
        max_feat = torch.amax(seed_features, dim=2)  # (B, C)
        avg_feat = torch.mean(seed_features, dim=2)  # (B, C)
        global_feat = torch.cat([max_feat, avg_feat], dim=1)  # (B, 2C)
        
        # Pose prediction
        pose_params = self.pose_head(global_feat)  # (B, 16)
        
        # Reshape to 4x4 pose matrix
        batch_size = x.shape[0]
        pose = pose_params.view(batch_size, 4, 4)  # (B, 4, 4)
        
        return pose


def project_to_se3_torch(T):
    """Differentiable projection to SE(3) without inplace ops on autograd views.
    
    优化：使用full_matrices=False只计算需要的奇异值（更高效）
    """
    # T: (B, 4, 4)
    R = T[:, :3, :3]
    # full_matrices=False: 只计算3x3矩阵需要的奇异值（更高效）
    U, _, Vh = torch.linalg.svd(R, full_matrices=False)
    # Build S to enforce det(U @ Vh) = +1
    B = R.shape[0]
    S = torch.eye(3, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    # 注意：必须使用torch.linalg.det（batched），不能用torch.det
    det_uv = torch.linalg.det(U @ Vh)
    neg_mask = det_uv < 0
    if neg_mask.any():
        S[neg_mask, 2, 2] = -1.0
    R_hat = U @ S @ Vh
    T_out = T.clone()
    T_out = T_out.to(dtype=R_hat.dtype)
    T_out[:, :3, :3] = R_hat
    # keep last row [0,0,0,1]
    last_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=T.device, dtype=R_hat.dtype).view(1, 1, 4)
    T_out[:, 3:4, :] = last_row
    return T_out


def _so3_geodesic_rad(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """计算SO(3)测地距离（弧度）"""
    R_delta = R_pred @ R_gt.transpose(1, 2)
    trace = torch.clamp((R_delta[:, 0, 0] + R_delta[:, 1, 1] + R_delta[:, 2, 2] - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(trace)

def _symmetry_aware_rot_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    对称性感知的旋转损失（处理钢筋交叉结构的对称性）
    
    对称性：
    1. 绕z轴旋转180°（两根钢筋互换）
    2. x轴或y轴翻转
    
    Args:
        R_pred: (B, 3, 3) 预测旋转矩阵
        R_gt: (B, 3, 3) GT旋转矩阵
    Returns:
        (B,) 每个样本的最小角度误差（考虑对称性）
    """
    # 基础误差
    theta_base = _so3_geodesic_rad(R_pred, R_gt)
    
    # 对称性1：绕z轴旋转180°（R_sym = R_gt @ Rz(180°)）
    # Rz(180°) = diag([-1, -1, 1])
    Rz_180 = torch.tensor([[-1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]], dtype=R_pred.dtype, device=R_pred.device).unsqueeze(0)
    R_gt_sym1 = R_gt @ Rz_180  # (B, 3, 3)
    theta_sym1 = _so3_geodesic_rad(R_pred, R_gt_sym1)
    
    # 对称性2：绕x轴翻转180°（Ry(180°)）
    Ry_180 = torch.tensor([[1, 0, 0],
                          [0, -1, 0],
                          [0, 0, -1]], dtype=R_pred.dtype, device=R_pred.device).unsqueeze(0)
    R_gt_sym2 = R_gt @ Ry_180
    theta_sym2 = _so3_geodesic_rad(R_pred, R_gt_sym2)
    
    # 对称性3：绕y轴翻转180°（Rx(180°)）
    Rx_180 = torch.tensor([[-1, 0, 0],
                          [0, 1, 0],
                          [0, 0, -1]], dtype=R_pred.dtype, device=R_pred.device).unsqueeze(0)
    R_gt_sym3 = R_gt @ Rx_180
    theta_sym3 = _so3_geodesic_rad(R_pred, R_gt_sym3)
    
    # 取最小误差（考虑所有对称性）
    theta_min = torch.minimum(theta_base, torch.minimum(theta_sym1, torch.minimum(theta_sym2, theta_sym3)))
    
    return theta_min

def get_pose_loss(end_points):
    pred_poses = end_points['pred_poses']
    gt_poses = end_points['gt_poses']
    # 投影预测到SE(3)
    pred_poses = project_to_se3_torch(pred_poses)
    # 分解
    R_pred, t_pred = pred_poses[:, :3, :3], pred_poses[:, :3, 3]
    R_gt,   t_gt   = gt_poses[:, :3, :3],   gt_poses[:, :3, 3]
    
    # 旋转损失：使用对称性感知损失（如果启用）或标准测地损失
    if cfgs.use_symmetry_loss:
        theta_rad = _symmetry_aware_rot_loss(R_pred, R_gt)
    else:
        theta_rad = _so3_geodesic_rad(R_pred, R_gt)
    rot_loss = theta_rad.mean()
    
    # 平移SmoothL1（L1稳定且鲁棒）
    trans_loss = F.smooth_l1_loss(t_pred, t_gt, reduction='none').sum(dim=1).mean()
    # 加权损失（平衡旋转和平移）
    loss = cfgs.w_rot * rot_loss + cfgs.w_trans * trans_loss
    end_points['loss/rot_loss'] = rot_loss
    end_points['loss/trans_loss'] = trans_loss
    end_points['loss/overall_loss'] = loss
    # 监控指标：角度(°)与平移(mm)
    end_points['metric/rot_deg_mean'] = (theta_rad * (180.0/np.pi)).mean()
    end_points['metric/trans_mm_mean'] = (t_pred - t_gt).norm(dim=1).mean() * 1000.0
    return loss, end_points


def train_one_epoch():
    """Train one epoch following official style"""
    global BATCH_TIMES
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step()  # decay BN momentum
    # set model to training mode
    net.train()
    
    # Create progress bar
    pbar = tqdm(enumerate(TRAIN_DATALOADER), 
                total=len(TRAIN_DATALOADER), 
                desc=f"Epoch {EPOCH_CNT+1}/{cfgs.max_epoch}",
                ncols=120,
                leave=True)
    
    epoch_start_time = time.time()
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    for batch_idx, batch_data_label in pbar:
        batch_start_time = time.time()
        
        # Move data to device (ensure float32 for pointnet2 ops)
        pointclouds = batch_data_label['pointcloud'].to(device, dtype=torch.float32)
        target_poses = batch_data_label['pose'].to(device, dtype=torch.float32)
        
        optimizer.zero_grad(set_to_none=True)
        # Forward with AMP
        # Disable AMP temporarily to avoid dtype issues in pointnet2 grouping
        with torch.cuda.amp.autocast(enabled=False):
            pred_poses = net(pointclouds)
            # Prepare end_points for loss calculation
            end_points = {
                'pred_poses': pred_poses,
                'gt_poses': target_poses
            }
            # Compute loss
            loss, end_points = get_pose_loss(end_points)
        
        # Backward + step with clipping
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        
        # Accumulate statistics and print out
        for key in end_points:
            if any(s in key for s in ['loss', 'acc', 'prec', 'recall', 'count', 'metric']):
                if key not in stat_dict: 
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        
        # Calculate time statistics
        batch_time = time.time() - batch_start_time
        BATCH_TIMES.append(batch_time)
        if len(BATCH_TIMES) > 100:  # Keep only recent 100 batch times
            BATCH_TIMES.pop(0)
        
        # Calculate remaining time
        avg_batch_time = np.mean(BATCH_TIMES) if BATCH_TIMES else 0
        remaining_batches = len(TRAIN_DATALOADER) - batch_idx - 1
        remaining_time = remaining_batches * avg_batch_time
        
        # Update progress bar
        current_lr = get_current_lr(EPOCH_CNT)
        overall_loss = stat_dict.get('loss/overall_loss', 0) / max(1, (batch_idx + 1))
        rot_deg = stat_dict.get('metric/rot_deg_mean', 0) / max(1, (batch_idx + 1))
        trans_mm = stat_dict.get('metric/trans_mm_mean', 0) / max(1, (batch_idx + 1))
        
        pbar.set_postfix({
            'Loss': f'{overall_loss:.4f}',
            'rot(deg)': f'{rot_deg:.2f}',
            't(mm)': f'{trans_mm:.1f}',
            'LR': f'{current_lr:.2e}',
            'ETA': format_time(remaining_time),
            'Batch/s': f'{1/avg_batch_time:.2f}' if avg_batch_time > 0 else '0.00'
        })
        
        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, 
                                     (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0
    
    pbar.close()
    
    # Record epoch time
    epoch_time = time.time() - epoch_start_time
    EPOCH_TIMES.append(epoch_time)
    if len(EPOCH_TIMES) > 5:  # Keep only recent 5 epoch times
        EPOCH_TIMES.pop(0)


def evaluate_one_epoch():
    """Evaluate one epoch following official style"""
    stat_dict = {}  # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    
    # Create evaluation progress bar
    pbar = tqdm(enumerate(TEST_DATALOADER), 
                total=len(TEST_DATALOADER), 
                desc="Evaluating",
                ncols=120,
                leave=False)
    
    for batch_idx, batch_data_label in pbar:
        # Move data to device
        pointclouds = batch_data_label['pointcloud'].to(device)
        target_poses = batch_data_label['pose'].to(device)
        
        # Forward pass
        with torch.no_grad():
            pred_poses = net(pointclouds)
            
            # Prepare end_points for loss calculation
            end_points = {
                'pred_poses': pred_poses,
                'gt_poses': target_poses
            }
            
            # Compute loss
            loss, end_points = get_pose_loss(end_points)
        
        # Accumulate statistics and print out
        for key in end_points:
            if any(s in key for s in ['loss', 'acc', 'prec', 'recall', 'count', 'metric']):
                if key not in stat_dict: 
                    stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        
        # Update progress bar
        if batch_idx > 0:
            current_loss = stat_dict.get('loss/overall_loss', 0) / (batch_idx + 1)
            pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
    
    pbar.close()
    
    num_batches = max(1, (batch_idx+1))
    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key]/float(num_batches), 
                             (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(num_batches))))
    
    mean_loss = stat_dict['loss/overall_loss']/float(num_batches)
    return mean_loss


def get_current_lr(epoch):
    """Get current learning rate following official style"""
    lr = cfgs.learning_rate
    for i, lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr


def adjust_learning_rate(optimizer, epoch):
    """Adjust learning rate following official style"""
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def format_time(seconds):
    """Format time display"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


# Create Dataset and Dataloader following official style
print("Creating dataset...")
TRAIN_DATASET = PoseDataset(cfgs.dataset_root, camera=cfgs.camera, split='train', 
                           num_points=cfgs.num_point, use_bbox=cfgs.use_bbox, 
                           bbox_size=cfgs.bbox_size,
                           use_yolo_bbox=cfgs.use_yolo_bbox)
TEST_DATASET = PoseDataset(cfgs.dataset_root, camera=cfgs.camera, split='test', 
                          num_points=cfgs.num_point, use_bbox=cfgs.use_bbox, 
                          bbox_size=cfgs.bbox_size,
                          use_yolo_bbox=cfgs.use_yolo_bbox)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(
    TRAIN_DATASET, batch_size=cfgs.batch_size, shuffle=True,
    num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn, pin_memory=True,
    persistent_workers=(cfgs.num_workers > 0)  # 仅在多进程时启用
)
TEST_DATALOADER = DataLoader(
    TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=cfgs.num_workers, worker_init_fn=my_worker_init_fn,
    collate_fn=collate_fn, pin_memory=True,
    persistent_workers=(cfgs.num_workers > 0)  # 仅在多进程时启用
)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))

# Init the model and optimizer following official style
net = PoseEstimationNet(input_feature_dim=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# 先用 dummy batch materialize LazyLinear 的 in_features（虚拟前向）
# 这样可以在加载checkpoint时使用strict=True（更严格）
with torch.no_grad():
    _dummy = torch.zeros(1, cfgs.num_point, 3, device=device, dtype=torch.float32)
    _ = net(_dummy)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)

# Load checkpoint if there is any
it = -1  # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    # 虚拟前向后可以使用strict=True（更严格），但保留False以兼容旧checkpoint
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))


def train(start_epoch):
    """Main training function following official style"""
    global EPOCH_CNT, TRAINING_START_TIME
    min_loss = 1e10
    loss = 0
    
    # Record training start time
    TRAINING_START_TIME = time.time()
    
    print(f"\n🚀 开始训练 6DoF Pose Estimation 模型")
    print(f"📊 训练配置:")
    print(f"   - 总epoch数: {cfgs.max_epoch}")
    print(f"   - 批次大小: {cfgs.batch_size}")
    print(f"   - 学习率: {cfgs.learning_rate}")
    print(f"   - 数据集: {cfgs.camera}")
    print(f"   - 使用bbox: {cfgs.use_bbox} (数据生成时已保存局部点云，通常不需要)")
    print(f"   - 使用YOLO bbox: {cfgs.use_yolo_bbox} (仅当use_bbox=True时有效)")
    print(f"   - 开始epoch: {start_epoch}")
    print("=" * 80)
    
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        
        # Calculate overall progress
        total_epochs = cfgs.max_epoch - start_epoch
        current_epoch = epoch - start_epoch + 1
        overall_progress = (current_epoch - 1) / total_epochs * 100
        
        # Calculate estimated completion time
        if EPOCH_TIMES:
            avg_epoch_time = np.mean(EPOCH_TIMES)
            remaining_epochs = cfgs.max_epoch - epoch
            estimated_remaining_time = remaining_epochs * avg_epoch_time
        else:
            estimated_remaining_time = 0
        
        # Calculate elapsed time
        elapsed_time = time.time() - TRAINING_START_TIME
        
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        
        print(f"\n📈 Epoch {epoch+1}/{cfgs.max_epoch} - 总体进度: {overall_progress:.1f}%")
        print(f"⏱️  已用时间: {format_time(elapsed_time)}")
        if estimated_remaining_time > 0:
            print(f"⏳ 预计剩余: {format_time(estimated_remaining_time)}")
        
        # Reset numpy seed
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()
        
        # Train one epoch
        train_one_epoch()
        
        # Evaluate
        print(f"\n🔍 开始评估...")
        loss = evaluate_one_epoch()
        
        # Save checkpoint
        save_dict = {'epoch': epoch+1,  # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try:  # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        
        # 保存最新checkpoint
        ckpt_latest = os.path.join(cfgs.log_dir, 'checkpoint.tar')
        torch.save(save_dict, ckpt_latest)
        
        # Display epoch summary
        epoch_time = EPOCH_TIMES[-1] if EPOCH_TIMES else 0
        print(f"\n✅ Epoch {epoch+1} 完成!")
        print(f"   - 训练时间: {format_time(epoch_time)}")
        print(f"   - 验证损失: {loss:.4f}")
        print(f"   - 最佳损失: {min_loss:.4f}")
        
        # 保存最佳模型
        if loss < min_loss:
            min_loss = loss
            ckpt_best = os.path.join(cfgs.log_dir, 'best_model.tar')
            torch.save(save_dict, ckpt_best)
            print(f"   🎉 新的最佳损失! 已保存到 {ckpt_best}")
        
        print("=" * 80)
    
    # Training completed
    total_time = time.time() - TRAINING_START_TIME
    print(f"\n🎉 训练完成!")
    print(f"⏱️  总训练时间: {format_time(total_time)}")
    print(f"📊 最终损失: {loss:.4f}")
    print(f"💾 模型已保存到: {cfgs.log_dir}/checkpoint.tar")


if __name__=='__main__':
    train(start_epoch)
