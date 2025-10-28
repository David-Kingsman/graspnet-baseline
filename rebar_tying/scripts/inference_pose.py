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


class PoseEstimationNet(nn.Module):
    """
    use GraspNet backbone for 6DoF pose estimation
    separate translation and rotation, add orthogonality constraint
    """
    
    def __init__(self, input_feature_dim=0, hidden_dim=256):
        super(PoseEstimationNet, self).__init__()
        
        # use GraspNet backbone
        self.backbone = Pointnet2Backbone(input_feature_dim)
        
        # separate prediction heads
        # translation prediction head (3D)
        self.translation_head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        
        # rotation prediction head (6D -> 3x3 matrix)
        self.rotation_head = nn.Sequential(
            nn.Linear(256 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # 6 parameters for building rotation matrix
        )
    
    def forward(self, x, end_points=None):
        """
        Args:
            x: (B, N, 3) point cloud
            end_points: additional end points information
        Returns:
            pose: (B, 4, 4) pose matrix
        """
        # extract features through backbone
        seed_features, seed_xyz, end_points = self.backbone(x, end_points)
        
        # global pooling
        max_feat = torch.max(seed_features, dim=2)[0]  # (B, 256)
        avg_feat = torch.mean(seed_features, dim=2)    # (B, 256)
        global_feat = torch.cat([max_feat, avg_feat], dim=1)  # (B, 512)
        
        # separate prediction heads for translation and rotation
        translation = self.translation_head(global_feat)  # (B, 3)
        rotation_params = self.rotation_head(global_feat)  # (B, 6)
        
        # build orthogonal rotation matrix from 6 parameters
        rotation_matrix = self._build_rotation_matrix(rotation_params)  # (B, 3, 3)
        
        # build 4x4 pose matrix
        batch_size = x.shape[0]
        pose = torch.eye(4, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose[:, :3, :3] = rotation_matrix
        pose[:, :3, 3] = translation
        
        return pose
    
    def _build_rotation_matrix(self, params):
        """
        build orthogonal rotation matrix from 6 parameters
        use Gram-Schmidt orthogonalization process
        """
        batch_size = params.shape[0]
        
        # reshape 6 parameters to 2 3D vectors
        v1 = params[:, :3]  # (B, 3)
        v2 = params[:, 3:]  # (B, 3)
        
        # Gram-Schmidt orthogonalization
        # normalize first vector
        u1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
        
        # orthogonalize second vector
        proj = torch.sum(u1 * v2, dim=1, keepdim=True)
        u2 = v2 - proj * u1
        u2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + 1e-8)
        
        # third vector obtained by cross product
        u3 = torch.cross(u1, u2, dim=1)
        
        # build rotation matrix
        rotation_matrix = torch.stack([u1, u2, u3], dim=1)  # (B, 3, 3)
        
        return rotation_matrix


def load_model(model_path, device='cuda'):
    """load trained model"""
    model = PoseEstimationNet(input_feature_dim=0, hidden_dim=256).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ loaded model: {model_path}")
    print(f"   best epoch: {checkpoint['epoch']}")
    print(f"   validation loss: {checkpoint['val_loss']:.6f}")
    
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


def evaluate_on_dataset(model, data_dir, num_samples=10, device='cuda'):
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
    
    # load point clouds and ground truth poses
    scene_folders = sorted(glob.glob(os.path.join(data_dir, 'scene_*')))[:1]  # only use first scene
    
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
            
            # predict pose
            pred_pose = predict_pose(model, pc, device)
            
            # calculate error
            trans_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
            rot_error = np.linalg.norm(pred_pose[:3, :3] @ gt_pose[:3, :3].T - np.eye(3))
            
            print(f"Frame {frame_idx}_obj{obj_idx}:")
            print(f"   translation error: {trans_error:.6f}m")
            print(f"   rotation error: {rot_error:.6f}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--pointcloud_path', help='Path to single pointcloud file')
    parser.add_argument('--data_dir', help='Path to dataset for evaluation')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to evaluate')
    
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
        evaluate_on_dataset(model, args.data_dir, args.num_samples, device)


if __name__ == '__main__':
    main()

