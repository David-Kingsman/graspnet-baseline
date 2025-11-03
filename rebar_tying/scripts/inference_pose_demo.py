#!/usr/bin/env python3
"""
use trained graspnet backbone to estimate 6DoF pose 
input: point cloud and output 4x4 pose matrix
evaluation: translation and rotation error

bash example:
python rebar_tying/scripts/inference_pose_demo.py --model_path rebar_tying/runs/6dof_pose_training/best_model.tar --data_dir rebar_tying/datasets/scenes/scene_incline --num_samples 3 --viz
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
    
    NOTE: This must match the architecture used in training (train_6dof_pose.py)
    """
    
    def __init__(self, input_feature_dim=0, hidden_dim=256):
        super(PoseEstimationNet, self).__init__()
        
        # use GraspNet backbone
        self.backbone = Pointnet2Backbone(input_feature_dim)
        
        # pose regression head - must match training architecture
        # includes BatchNorm and higher Dropout for regularization
        self.pose_head = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
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
    
    # Materialize LazyLinear first (same as training)
    with torch.no_grad():
        model.eval()
        _dummy = torch.zeros(1, 20000, 3, device=device, dtype=torch.float32)
        _ = model(_dummy)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load with strict=False first to check compatibility
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"⚠️  Warning: strict loading failed, trying with strict=False")
        print(f"   Error: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model.eval()
    
    print(f"✅ loaded model: {model_path}")
    if 'epoch' in checkpoint:
        print(f"   epoch: {checkpoint['epoch']}")
    if 'loss' in checkpoint:
        print(f"   loss: {checkpoint['loss']:.6f}")
    
    return model, checkpoint


def predict_pose(model, pointcloud, device='cuda', max_points=20000, random_seed=None):
    """
    predict pose for single point cloud
    
    Args:
        model: trained model
        pointcloud: (N, 3) numpy array 或 (1, N, 3) torch tensor
        device: device
        max_points: maximum number of points (same as training)
        random_seed: random seed for reproducible sampling (None for random)
        
    Returns:
        pose: (4, 4) numpy array - pose matrix
    """
    model.eval()
    
    # convert to numpy if needed (process in numpy for consistency with training)
    if isinstance(pointcloud, torch.Tensor):
        pointcloud = pointcloud.cpu().numpy()
    
    # ensure it's 2D (N, 3)
    if pointcloud.ndim == 3:
        pointcloud = pointcloud[0]  # take first batch
    
    # downsample to fixed size (same as training: use np.random.choice)
    # Set random seed for reproducibility if provided
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    if len(pointcloud) > max_points:
        indices = rng.choice(len(pointcloud), max_points, replace=False)
        pointcloud = pointcloud[indices]
    elif len(pointcloud) < max_points:
        indices = rng.choice(len(pointcloud), max_points, replace=True)
        pointcloud = pointcloud[indices]
    
    # convert to torch tensor for model
    pointcloud = torch.FloatTensor(pointcloud).unsqueeze(0)  # (1, N, 3)
    
    # center
    center = pointcloud.mean(dim=1, keepdim=True)  # (1, 1, 3)
    pointcloud = pointcloud - center
    
    # model inference
    with torch.no_grad():
        pointcloud = pointcloud.to(device)
        pred_pose = model(pointcloud)
        
        # restore translation (because it was centered during training)
        pred_pose = pred_pose.cpu().numpy()[0]  # (4, 4)
        center_vec = center.cpu().numpy()[0, 0, :]  # (3,) - get the 3D center vector
        pred_pose[:3, 3] += center_vec  # add original center
    
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


def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray, use_symmetry: bool = True, verbose: bool = False) -> float:
    """
    Compute rotation error in degrees using geodesic distance.
    
    Args:
        R_pred: (3, 3) predicted rotation matrix
        R_gt: (3, 3) ground truth rotation matrix
        use_symmetry: if True, consider symmetry-aware error (same as training)
        verbose: if True, print detailed error information for debugging
    
    Returns:
        rotation error in degrees
    """
    def _so3_geodesic_rad_np(R_pred, R_gt):
        """Compute SO(3) geodesic distance in radians"""
        R_delta = R_pred @ R_gt.T
        trace = np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0)
        return math.acos(trace)
    
    # Base error
    theta_base = _so3_geodesic_rad_np(R_pred, R_gt)
    theta_base_deg = theta_base * 180.0 / math.pi
    
    if not use_symmetry:
        return theta_base_deg
    
    # Symmetry-aware error (same as training)
    # Symmetry 1: rotate 180° around z-axis
    Rz_180 = np.array([[-1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]], dtype=R_gt.dtype)
    R_gt_sym1 = R_gt @ Rz_180
    theta_sym1 = _so3_geodesic_rad_np(R_pred, R_gt_sym1)
    theta_sym1_deg = theta_sym1 * 180.0 / math.pi
    
    # Symmetry 2: flip around x-axis (Ry(180°))
    Ry_180 = np.array([[1, 0, 0],
                      [0, -1, 0],
                      [0, 0, -1]], dtype=R_gt.dtype)
    R_gt_sym2 = R_gt @ Ry_180
    theta_sym2 = _so3_geodesic_rad_np(R_pred, R_gt_sym2)
    theta_sym2_deg = theta_sym2 * 180.0 / math.pi
    
    # Symmetry 3: flip around y-axis (Rx(180°))
    Rx_180 = np.array([[-1, 0, 0],
                      [0, 1, 0],
                      [0, 0, -1]], dtype=R_gt.dtype)
    R_gt_sym3 = R_gt @ Rx_180
    theta_sym3 = _so3_geodesic_rad_np(R_pred, R_gt_sym3)
    theta_sym3_deg = theta_sym3 * 180.0 / math.pi
    
    # Take minimum error (considering all symmetries)
    errors = [theta_base, theta_sym1, theta_sym2, theta_sym3]
    theta_min = min(errors)
    theta_min_deg = theta_min * 180.0 / math.pi
    
    # Debug output if verbose or if error is large
    if verbose or theta_min_deg > 30.0:
        print(f"      Rotation error breakdown:")
        print(f"        - Base (no symmetry): {theta_base_deg:.2f}°")
        print(f"        - Sym1 (z-axis 180°): {theta_sym1_deg:.2f}°")
        print(f"        - Sym2 (y-axis flip): {theta_sym2_deg:.2f}°")
        print(f"        - Sym3 (x-axis flip): {theta_sym3_deg:.2f}°")
        print(f"        - Min (symmetry-aware): {theta_min_deg:.2f}°")
    
    return theta_min_deg


def draw_axes_o3d(T: np.ndarray, length: float = 0.1, style: str = "normal"):
    """
    Draw coordinate axes (R/G/B = X/Y/Z)
    
    Args:
        T: 4x4 pose matrix
        length: axis length
        style: "gt" (thin lines + green sphere), "pred" (thick lines + red sphere), or "normal" (no keypoint)
    """
    if o3d is None:
        return []
    origin = T[:3, 3]
    Rx = T[:3, 0] * length
    Ry = T[:3, 1] * length
    Rz = T[:3, 2] * length
    
    # RGB for XYZ (standard)
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Red=X, Green=Y, Blue=Z
    directions = [Rx, Ry, Rz]
    
    result = []
    
    # Draw axes with different line thickness
    line_radius = 0.0005 if style == "gt" else 0.0015  # Thin for GT, thick for Pred
    
    for i, (dir_vec, color) in enumerate(zip(directions, colors)):
        # Create cylinder for each axis
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=line_radius, height=np.linalg.norm(dir_vec))
        
        # Rotate cylinder to point in the right direction
        z_axis = np.array([0, 0, 1])
        target_dir = dir_vec / (np.linalg.norm(dir_vec) + 1e-9)
        
        # Compute rotation
        if np.abs(np.dot(z_axis, target_dir)) > 0.99:
            # Already aligned or opposite, use identity or 180 rotation
            if np.dot(z_axis, target_dir) < 0:
                R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
            else:
                R = np.eye(3)
        else:
            # General case: rotate around axis perpendicular to both
            axis = np.cross(z_axis, target_dir)
            axis = axis / (np.linalg.norm(axis) + 1e-9)
            angle = np.arccos(np.clip(np.dot(z_axis, target_dir), -1.0, 1.0))
            # Simple rotation matrix using cross product formula
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        cyl.rotate(R, center=[0, 0, 0])
        
        # Translate to origin
        cyl.translate(origin + dir_vec / 2)
        cyl.paint_uniform_color(color)
        result.append(cyl)
    
    # Add origin sphere (keypoint) with different color for GT vs Pred
    if style in ["gt", "pred"]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)  # Smaller sphere
        sphere.translate(origin)
        # GT: green sphere, Pred: red sphere
        sphere.paint_uniform_color([0, 1, 0] if style == "gt" else [1, 0, 0])
        result.append(sphere)
    
    return result


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
        
        for idx, pc_file in enumerate(pc_files):
            try:
                # load point cloud
                pc = np.load(pc_file)
                basename = os.path.basename(pc_file)
                parts = basename.replace('.npy', '').split('_')
                if len(parts) < 2:
                    print(f"Warning: Invalid filename format: {basename}, skipping...")
                    continue
                frame_idx = int(parts[0])
                obj_idx = int(parts[1].replace('obj', ''))
                
                # load ground truth pose
                meta_file = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
                if not os.path.exists(meta_file):
                    print(f"Warning: Meta file not found: {meta_file}, skipping...")
                    continue
                
                meta = sio.loadmat(meta_file)
                poses = meta['poses']
                if obj_idx >= len(poses):
                    print(f"Warning: obj_idx {obj_idx} >= {len(poses)} in {meta_file}, skipping...")
                    continue
                gt_pose = poses[obj_idx]
            except Exception as e:
                print(f"Error processing {pc_file}: {e}, skipping...")
                continue
            
            # predict pose and project to SE(3)
            # Use frame_idx as seed for reproducible results
            pred_pose = predict_pose(model, pc, device, random_seed=frame_idx)
            pred_pose = project_to_se3(pred_pose)
            
            # calculate error
            # Use symmetry-aware rotation error (same as training)
            trans_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
            rot_error = rotation_error_deg(pred_pose[:3, :3], gt_pose[:3, :3], use_symmetry=True, verbose=(idx < 3))
            
            # Also compute raw error (without symmetry) for comparison
            rot_error_raw = rotation_error_deg(pred_pose[:3, :3], gt_pose[:3, :3], use_symmetry=False)
            
            print(f"Frame {frame_idx}_obj{obj_idx}:")
            print(f"   translation error: {trans_error*1000:.2f} mm")
            print(f"   rotation error (symmetry-aware): {rot_error:.2f} deg")
            
            # Always show raw error if symmetry-aware error is large (>30 deg)
            # This helps diagnose issues
            if rot_error > 30.0 or abs(rot_error_raw - rot_error) > 5.0:
                print(f"   rotation error (raw, no symmetry): {rot_error_raw:.2f} deg")
                if abs(rot_error_raw - rot_error) > 5.0:
                    print(f"   → Note: prediction may be a symmetric pose (error reduced by {rot_error_raw - rot_error:.1f}°)")
                if rot_error > 30.0:
                    print(f"   ⚠️  Warning: Large rotation error detected! This may indicate:")
                    print(f"      - Prediction is significantly off from GT")
                    print(f"      - Symmetry may not fully account for this error")
                    print(f"      - Check if this sample is in the training distribution")
            rows.append([scene_name, frame_idx, obj_idx, trans_error, rot_error])

            # optional visualization with interactive browsing
            if viz and o3d is not None:
                import numpy as _np
                import open3d as _o3d
                pcd = _o3d.geometry.PointCloud()
                pcd.points = _o3d.utility.Vector3dVector(pc.astype(_np.float64))
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                
                # Find the best symmetric pose for visualization (closest to GT)
                pred_pose_viz = pred_pose.copy()
                R_pred = pred_pose[:3, :3]
                R_gt = gt_pose[:3, :3]
                
                # Check all symmetric poses and use the one closest to GT
                def _so3_geodesic_rad_np(R1, R2):
                    R_delta = R1 @ R2.T
                    trace = np.clip((np.trace(R_delta) - 1) / 2.0, -1.0, 1.0)
                    return math.acos(trace)
                
                poses_to_check = [pred_pose]  # Original
                
                # Add symmetric poses
                Rz_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=R_pred.dtype)
                Ry_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=R_pred.dtype)
                Rx_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=R_pred.dtype)
                
                sym_transforms = [
                    Rz_180,
                    Ry_180,
                    Rx_180
                ]
                
                for sym_R in sym_transforms:
                    sym_pose = pred_pose.copy()
                    sym_pose[:3, :3] = R_pred @ sym_R
                    poses_to_check.append(sym_pose)
                
                # Find pose with minimum error to GT
                best_error = float('inf')
                for pose in poses_to_check:
                    err = _so3_geodesic_rad_np(pose[:3, :3], R_gt)
                    if err < best_error:
                        best_error = err
                        pred_pose_viz = pose
                
                # GT in green, Pred (best symmetric) in red with keypoint spheres
                geoms = [pcd] + draw_axes_o3d(gt_pose, 0.05, style="gt") + draw_axes_o3d(pred_pose_viz, 0.05, style="pred")
                
                # Use key callbacks for interactive navigation
                quit_flag = [False]
                
                def on_c(vis):
                    print("  → Continuing to next sample...")
                    return False  # Continue to next
                
                def on_q(vis):
                    print("\n  → Quitting visualization...")
                    quit_flag[0] = True
                    return True  # Close window
                
                key_to_callback = {
                    ord('C'): on_c,
                    ord('c'): on_c,
                    ord('Q'): on_q,
                    ord('q'): on_q,
                    256: on_q  # ESC
                }
                
                print(f"\n  [{idx+1}/{len(pc_files)}] Press 'C' to continue, 'Q' to quit")
                
                _o3d.visualization.draw_geometries_with_key_callbacks(
                    geoms,
                    key_to_callback,
                    window_name=f"{scene_name}:{frame_idx}_obj{obj_idx} ({idx+1}/{len(pc_files)})"
                )
                
                # Check if user wants to quit entirely
                if quit_flag[0]:
                    break

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
        pred_pose = project_to_se3(pred_pose)  # project to valid SE(3)
        
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

