#!/usr/bin/env python3
"""
use trained graspnet backbone to estimate 6DoF pose 
input: point cloud and output 4x4 pose matrix
evaluation: translation and rotation error

bash example:
python inference_pose.py --model_path /path/to/model.pth --pointcloud_path /path/to/pointcloud.npy --data_dir /path/to/dataset
python inference_pose.py --model_path /path/to/model.pth --data_dir /path/to/dataset --num_samples 10
"""

import os
import sys
import numpy as np
import math
import csv
import torch
import torch.nn as nn
import argparse
import glob
import scipy.io as sio

# add GraspNet path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from backbone import Pointnet2Backbone

try:
    import open3d as o3d
except Exception:
    o3d = None


class PoseEstimationNet(nn.Module):
    """
    use GraspNet backbone for 6DoF pose estimation
    separate translation and rotation, add orthogonality constraint
    """
    
    def __init__(self, input_feature_dim=0, hidden_dim=256):
        super(PoseEstimationNet, self).__init__()
        
        # use GraspNet backbone
        self.backbone = Pointnet2Backbone(input_feature_dim)
        
        # pose regression head
        # backbone output (B, 1024, num_seed) features
        # we need to pool to get global features
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # pose prediction head with LazyLinear to adapt 2*C channels
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
        # 注意：本工程backbone期望输入(B, N, 3)，无需转置
        # extract features through backbone
        seed_features, seed_xyz, end_points = self.backbone(x, end_points)
        
        # global pooling
        max_feat = torch.amax(seed_features, dim=2)  # (B, C)
        avg_feat = torch.mean(seed_features, dim=2)  # (B, C)
        global_feat = torch.cat([max_feat, avg_feat], dim=1)  # (B, 2C)
        
        # pose prediction
        pose_params = self.pose_head(global_feat)  # (B, 16)
        
        # reshape to 4x4 pose matrix
        batch_size = x.shape[0]
        pose = pose_params.view(batch_size, 4, 4)  # (B, 4, 4)
        
        return pose
    


def load_model(model_path, device='cuda'):
    """load trained model"""
    model = PoseEstimationNet(input_feature_dim=0, hidden_dim=256).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ loaded model: {model_path}")
    if 'epoch' in checkpoint:
        print(f"   epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   loss: {checkpoint['loss']:.6f}")
    
    return model, checkpoint


def predict_pose(model, pointcloud, device='cuda', max_points=20000):
    """
    predict pose for single point cloud
    
    Args:
        model: trained model
        pointcloud: (N, 3) numpy array 或 (1, N, 3) torch tensor
        device: device
        max_points: maximum number of points (same as training)
        
    Returns:
        pose: (4, 4) numpy array - pose matrix
    """
    model.eval()
    
    # convert to torch tensor
    if isinstance(pointcloud, np.ndarray):
        pointcloud = torch.FloatTensor(pointcloud).unsqueeze(0)  # (1, N, 3)
    
    # downsample to fixed size (same as training)
    if pointcloud.shape[1] > max_points:
        indices = torch.randperm(pointcloud.shape[1])[:max_points]
        pointcloud = pointcloud[:, indices, :]
    elif pointcloud.shape[1] < max_points:
        indices = torch.randint(0, pointcloud.shape[1], (max_points,))
        pointcloud = pointcloud[:, indices, :]
    
    # center
    center = pointcloud.mean(dim=1, keepdim=True)  # (1, 1, 3)
    pointcloud = pointcloud - center
    
    # model inference
    with torch.no_grad():
        pointcloud = pointcloud.to(device)
        pred_pose = model(pointcloud)
        
        # restore translation (because it was centered during training)
        pred_pose = pred_pose.cpu().numpy()[0]  # (4, 4)
        pred_pose[:3, 3] += center.cpu().numpy()[0, 0]  # add original center
    
    return pred_pose


def project_to_se3(pose4x4: np.ndarray) -> np.ndarray:
    """Project arbitrary 4x4 to SE(3): R via SVD (orthogonal, det=+1), last row [0,0,0,1]."""
    T = pose4x4.copy()
    R0 = T[:3, :3]
    U, _, Vt = np.linalg.svd(R0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    T[:3, :3] = R
    T[3, :] = np.array([0, 0, 0, 1], dtype=T.dtype)
    return T


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Compute rotation error in degrees using geodesic distance."""
    R_delta = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0)
    theta = math.acos(trace)
    return theta * 180.0 / math.pi


def draw_axes_o3d(T: np.ndarray, length: float = 0.1):
    if o3d is None:
        return []
    origin = T[:3, 3]
    Rx = T[:3, 0] * length
    Ry = T[:3, 1] * length
    Rz = T[:3, 2] * length
    points = [origin, origin + Rx, origin, origin + Ry, origin, origin + Rz]
    lines = [[0, 1], [2, 3], [4, 5]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return [ls]


def evaluate_on_dataset(model, data_dir, num_samples=10, device='cuda', save_csv: str = None, viz: bool = False):
    """
    evaluate model on test data
    
    Args:
        model: trained model
        data_dir: data directory
        num_samples: number of samples to evaluate
        device: device
    """
    print("\n" + "="*80)
    print("start evaluating model")
    print("="*80)
    
    # determine scenes: allow passing either datasets root or a specific scene
    if os.path.isdir(os.path.join(data_dir, 'pointclouds')):
        scene_folders = [data_dir]
    else:
        scene_folders = sorted(glob.glob(os.path.join(data_dir, 'scene_*')))

    rows = []
    
    for scene_path in scene_folders:
        scene_name = os.path.basename(scene_path)
        pc_dir = os.path.join(scene_path, 'pointclouds')
        
        if not os.path.exists(pc_dir):
            continue
        
        pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.npy')))[:num_samples]
        
        print(f"\nscene: {scene_name}")
        print(f"evaluate {len(pc_files)} samples\n")
        
        for pc_file in pc_files:
            # load point cloud
            pc = np.load(pc_file)
            basename = os.path.basename(pc_file)
            parts = basename.replace('.npy', '').split('_')
            frame_idx = int(parts[0])
            obj_idx = int(parts[1].replace('obj', ''))
            
            # load ground truth pose
            meta_file = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
            if not os.path.exists(meta_file):
                continue
            
            meta = sio.loadmat(meta_file)
            gt_pose = meta['poses'][obj_idx]
            
            # predict pose and project to SE(3)
            pred_pose = predict_pose(model, pc, device)
            pred_pose = project_to_se3(pred_pose)
            
            # calculate error
            trans_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
            rot_error = rotation_error_deg(pred_pose[:3, :3], gt_pose[:3, :3])
            
            print(f"Frame {frame_idx}_obj{obj_idx}:")
            print(f"   translation error: {trans_error*1000:.2f} mm")
            print(f"   rotation error: {rot_error:.2f} deg")
            rows.append([scene_name, frame_idx, obj_idx, trans_error, rot_error])

            # optional visualization
            if viz and o3d is not None:
                import numpy as _np
                import open3d as _o3d
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pc.astype(_np.float64))
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                geoms = [pcd] + draw_axes_o3d(gt_pose, 0.05) + draw_axes_o3d(pred_pose, 0.05)
                _o3d.visualization.draw_geometries(geoms, window_name=f"{scene_name}:{frame_idx}_obj{obj_idx}")

    # summary
    if rows:
        trans_mm = [r[3]*1000.0 for r in rows]
        rot_deg = [r[4] for r in rows]
        mean_trans = float(np.mean(trans_mm))
        median_trans = float(np.median(trans_mm))
        mean_rot = float(np.mean(rot_deg))
        median_rot = float(np.median(rot_deg))
        print("\nSummary (on evaluated samples):")
        print(f"  Translation error: mean {mean_trans:.2f} mm | median {median_trans:.2f} mm")
        print(f"  Rotation error:    mean {mean_rot:.2f} deg | median {median_rot:.2f} deg")

        if save_csv:
            os.makedirs(os.path.dirname(save_csv), exist_ok=True) if os.path.dirname(save_csv) else None
            with open(save_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['scene', 'frame_idx', 'obj_idx', 'trans_error_m', 'rot_error_deg'])
                writer.writerows(rows)
            print(f"  Saved CSV: {save_csv}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--pointcloud_path', help='Path to single pointcloud file')
    parser.add_argument('--data_dir', help='Path to dataset for evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--viz', action='store_true', help='Visualize point cloud with GT/Pred axes')
    parser.add_argument('--save_csv', default=None, help='Path to save evaluation CSV')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model
    model, checkpoint = load_model(args.model_path, device)
    
    # inference mode 1: single point cloud prediction
    if args.pointcloud_path:
        pc = np.load(args.pointcloud_path)
        pred_pose = predict_pose(model, pc, device)
        
        print("\n" + "="*80)
        print("predicted results:")
        print("="*80)
        print("predicted pose matrix:")
        print(pred_pose)
        print("\n rotation matrix R:")
        print(pred_pose[:3, :3])
        print("\n translation vector t:")
        print(pred_pose[:3, 3])
    
    # inference mode 2: dataset evaluation
    if args.data_dir:
        evaluate_on_dataset(model, args.data_dir, args.num_samples, device, save_csv=args.save_csv, viz=args.viz)


if __name__ == '__main__':
    main()

