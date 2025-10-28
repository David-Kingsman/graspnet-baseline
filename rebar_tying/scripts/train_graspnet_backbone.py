#!/usr/bin/env python3
"""
Training script using GraspNet backbone for 6DoF pose estimation
"""

import os
import sys
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob
import cv2
import scipy.io as sio
import time
from datetime import datetime

# path to the root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from backbone import Pointnet2Backbone
from pytorch_utils import BNMomentumScheduler


class TyingPoseDataset(Dataset):
    """
    dataset for 6DoF pose estimation with GraspNet backbone format (pointcloud and pose)
    """
    
    def __init__(self, scenes_dir, max_points=20000, use_augment=False):
        """
        Args:
            scenes_dir: directory of scenes
            max_points: maximum number of points after downsampling
            use_augment: whether to use data augmentation
        """
        self.scenes_dir = scenes_dir
        self.max_points = max_points
        self.use_augment = use_augment
        
        # get all scenes
        scene_folders = sorted(glob.glob(os.path.join(scenes_dir, 'scene_*')))
        print(f"found {len(scene_folders)} scenes")
        
        # build sample list
        self.samples = []
        for scene_path in scene_folders:
            scene_name = os.path.basename(scene_path)
            
            # get all pointclouds files in the pointclouds directory
            pc_dir = os.path.join(scene_path, 'pointclouds')
            if not os.path.exists(pc_dir):
                continue
                
            pc_files = sorted(glob.glob(os.path.join(pc_dir, '*.npy')))
            
            for pc_file in pc_files:
                basename = os.path.basename(pc_file)
                # format: XXXX_objY.npy
                parts = basename.replace('.npy', '').split('_')
                frame_idx = int(parts[0])
                obj_idx = int(parts[1].replace('obj', ''))
                
                # load corresponding meta file to get pose
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
        
        print(f"total {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # load pointcloud
        pc = np.load(sample['pc_file']).astype(np.float32)
        
        # data augmentation (if needed)
        if self.use_augment:
            pc = self._augment_pointcloud(pc)
        
        # random sample to fixed size
        pc = self._downsample_pointcloud(pc, self.max_points)
        
        # normalize
        center = np.mean(pc, axis=0)
        pc = pc - center
        
        # load pose (4x4 matrix)
        pose = sample['pose'].astype(np.float32)
        
        # convert to torch tensor
        pc = torch.FloatTensor(pc)
        pose = torch.FloatTensor(pose)
        
        return {
            'pointcloud': pc,
            'pose': pose,
            'scene': sample['scene'],
            'frame_idx': sample['frame_idx'],
            'obj_idx': sample['obj_idx']
        }
    
    def _downsample_pointcloud(self, pc, max_points):
        """downsample pointcloud to fixed size"""
        if len(pc) > max_points:
            # random sampling
            indices = np.random.choice(len(pc), max_points, replace=False)
            pc = pc[indices]
        elif len(pc) < max_points:
            # repeat sampling
            indices = np.random.choice(len(pc), max_points, replace=True)
            pc = pc[indices]
        return pc
    
    def _augment_pointcloud(self, pc):
        """data augmentation: add noise"""
        # add small gaussian noise
        noise = np.random.randn(*pc.shape) * 0.01
        pc = pc + noise
        return pc


def collate_fn(batch):
    """Custom collate for batches"""
    pointclouds = [item['pointcloud'] for item in batch]
    poses = [item['pose'] for item in batch]
    
    # ensure all pointclouds have the same shape
    max_points = max(pc.shape[0] for pc in pointclouds)
    
    padded_pcs = []
    for pc in pointclouds:
        padded = torch.zeros(max_points, pc.shape[1])
        padded[:pc.shape[0]] = pc
        padded_pcs.append(padded)
    
    pointclouds = torch.stack(padded_pcs)
    poses = torch.stack(poses)
    
    return {
        'pointcloud': pointclouds,
        'pose': poses
    }

def my_worker_init_fn(worker_id):
    """worker initialization function, set random seed"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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
        
        # separate prediction head
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
            nn.Linear(256, 6)  # 6个参数用于构建旋转矩阵
        )
    
    def forward(self, x, end_points=None):
        """
        Args:
            x: (B, N, 3) pointcloud
            end_points: additional end point information
        Returns:
            pose: (B, 4, 4) pose matrix
        """
        # extract features through backbone
        seed_features, seed_xyz, end_points = self.backbone(x, end_points)
        
        # global pooling
        max_feat = torch.max(seed_features, dim=2)[0]  # (B, 256)
        avg_feat = torch.mean(seed_features, dim=2)    # (B, 256)
        global_feat = torch.cat([max_feat, avg_feat], dim=1)  # (B, 512)
        
        # separate prediction for translation and rotation
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
        using Gram-Schmidt orthogonalization process
        """
        batch_size = params.shape[0]
        
        # reshape 6 parameters to 2 3D vectors
        v1 = params[:, :3]  # (B, 3)
        v2 = params[:, 3:]  # (B, 3)
        
        # Gram-Schmidt orthogonalization
        # first vector normalization
        u1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + 1e-8)
        
        # second vector orthogonalization
        proj = torch.sum(u1 * v2, dim=1, keepdim=True)
        u2 = v2 - proj * u1
        u2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + 1e-8)
        
        # third vector obtained by cross product
        u3 = torch.cross(u1, u2, dim=1)
        
        # build rotation matrix
        rotation_matrix = torch.stack([u1, u2, u3], dim=1)  # (B, 3, 3)
        
        return rotation_matrix


def pose_loss(pred_pose, gt_pose, trans_weight=1.0, rot_weight=20.0, ortho_weight=5.0):
    """
    improved 6DoF pose loss function
    separate translation and rotation loss, add orthogonality constraint
    """
    batch_size = pred_pose.shape[0]
    
    # translation loss (L2 loss on translation)
    pred_trans = pred_pose[:, :3, 3]  # (B, 3)
    gt_trans = gt_pose[:, :3, 3]       # (B, 3)
    trans_loss = torch.mean(torch.norm(pred_trans - gt_trans, dim=1))
    
    # rotation loss (Frobenius norm)
    pred_rot = pred_pose[:, :3, :3]   # (B, 3, 3)
    gt_rot = gt_pose[:, :3, :3]       # (B, 3, 3)
    
    # calculate relative rotation error
    rot_diff = torch.bmm(pred_rot, gt_rot.transpose(-2, -1))  # (B, 3, 3)
    identity = torch.eye(3, device=rot_diff.device).unsqueeze(0).expand(batch_size, -1, -1)
    rot_loss = torch.mean(torch.norm(rot_diff - identity, dim=(1, 2)))
    
    # orthogonality constraint loss
    # ensure predicted rotation matrix is orthogonal
    ortho_loss = torch.mean(torch.norm(torch.bmm(pred_rot, pred_rot.transpose(-2, -1)) - identity, dim=(1, 2)))
    
    # combine loss
    total_loss = trans_weight * trans_loss + rot_weight * rot_loss + ortho_weight * ortho_loss
    
    return total_loss, trans_loss, rot_loss, ortho_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """train one epoch"""
    model.train()
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    total_ortho_loss = 0
    num_batches = 0
    
    # create progress bar
    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc=f"Epoch {epoch+1}",
                ncols=120,
                leave=True)
    
    epoch_start_time = time.time()
    
    for batch_idx, batch in pbar:
        batch_start_time = time.time()
        
        pointclouds = batch['pointcloud'].to(device)
        target_poses = batch['pose'].to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        pred_poses = model(pointclouds)
        
        # calculate loss
        loss, trans_loss, rot_loss, ortho_loss = criterion(pred_poses, target_poses)
        
        # backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_trans_loss += trans_loss.item()
        total_rot_loss += rot_loss.item()
        total_ortho_loss += ortho_loss.item()
        num_batches += 1
        
        # calculate time statistics
        batch_time = time.time() - batch_start_time
        
        # update progress bar
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / num_batches
        
        pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{current_lr:.2e}',
            'Batch/s': f'{1/batch_time:.2f}' if batch_time > 0 else '0.00'
        })
        
        # record to TensorBoard
        if writer and batch_idx % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/TransLoss', trans_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/RotLoss', rot_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/OrthoLoss', ortho_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/LearningRate', current_lr, epoch * len(dataloader) + batch_idx)
    
    pbar.close()
    
    return (total_loss / num_batches,
            total_trans_loss / num_batches,
            total_rot_loss / num_batches,
            total_ortho_loss / num_batches)


def validate(model, dataloader, criterion, device, epoch, writer=None):
    """validate model"""
    model.eval()
    total_loss = 0
    total_trans_loss = 0
    total_rot_loss = 0
    total_ortho_loss = 0
    num_batches = 0
    
    # create validation progress bar
    pbar = tqdm(enumerate(dataloader), 
                total=len(dataloader), 
                desc="Validating",
                ncols=120,
                leave=False)
    
    with torch.no_grad():
        for batch_idx, batch in pbar:
            pointclouds = batch['pointcloud'].to(device)
            target_poses = batch['pose'].to(device)
            
            # forward pass
            pred_poses = model(pointclouds)
            
            # calculate loss
            loss, trans_loss, rot_loss, ortho_loss = criterion(pred_poses, target_poses)
            
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
            total_ortho_loss += ortho_loss.item()
            num_batches += 1
            
            # update progress bar
            if batch_idx > 0:
                current_loss = total_loss / num_batches
                pbar.set_postfix({'Loss': f'{current_loss:.4f}'})
    
    pbar.close()
    
    # record to TensorBoard
    if writer:
        writer.add_scalar('Val/Loss', total_loss / num_batches, epoch)
        writer.add_scalar('Val/TransLoss', total_trans_loss / num_batches, epoch)
        writer.add_scalar('Val/RotLoss', total_rot_loss / num_batches, epoch)
        writer.add_scalar('Val/OrthoLoss', total_ortho_loss / num_batches, epoch)
    
    return (total_loss / num_batches,
            total_trans_loss / num_batches,
            total_rot_loss / num_batches,
            total_ortho_loss / num_batches)


def get_current_lr(epoch, initial_lr, lr_decay_steps, lr_decay_rates):
    """get current learning rate"""
    lr = initial_lr
    for i, lr_decay_epoch in enumerate(lr_decay_steps):
        if epoch >= lr_decay_epoch:
            lr *= lr_decay_rates[i]
    return lr

def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_steps, lr_decay_rates):
    """adjust learning rate"""
    lr = get_current_lr(epoch, initial_lr, lr_decay_steps, lr_decay_rates)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def format_time(seconds):
    """format time display"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"

def parse_args():
    parser = argparse.ArgumentParser(description='Train 6DoF pose with GraspNet backbone')
    parser.add_argument('--data_dir', default='rebar_tying/datasets/scenes',
                       help='Scenes directory')
    parser.add_argument('--epoch', type=int, default=50, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--output_dir', default='rebar_tying/runs/graspnet_backbone',
                       help='Output directory')
    parser.add_argument('--max_points', type=int, default=20000, help='Max points per cloud')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--checkpoint_path', default=None, help='Checkpoint path to resume training')
    parser.add_argument('--bn_decay_step', type=int, default=2, help='BN decay step')
    parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='BN decay rate')
    parser.add_argument('--lr_decay_steps', default='20,35,45', help='LR decay steps')
    parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='LR decay rates')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # parse learning rate decay parameters
    lr_decay_steps = [int(x) for x in args.lr_decay_steps.split(',')]
    lr_decay_rates = [float(x) for x in args.lr_decay_rates.split(',')]
    assert len(lr_decay_steps) == len(lr_decay_rates)
    
    print("="*80)
    print("Training 6DoF Pose with GraspNet Backbone (Enhanced)")
    print("="*80)
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # create TensorBoard writers
    train_writer = SummaryWriter(os.path.join(args.output_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(args.output_dir, 'val'))
    
    # create dataset
    print(f"\n1. Loading dataset from: {args.data_dir}")
    dataset = TyingPoseDataset(args.data_dir, max_points=args.max_points, use_augment=args.augment)
    
    # split dataset (80/20)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, total_size))
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                             shuffle=True, collate_fn=collate_fn, 
                             num_workers=args.num_workers, worker_init_fn=my_worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, collate_fn=collate_fn, 
                           num_workers=args.num_workers, worker_init_fn=my_worker_init_fn)
    
    # create model
    print(f"\n2. Creating GraspNet backbone model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    model = PoseEstimationNet(input_feature_dim=0, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = pose_loss
    
    # BN momentum scheduler
    BN_MOMENTUM_INIT = 0.5
    BN_MOMENTUM_MAX = 0.001
    bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * args.bn_decay_rate**(int(it / args.bn_decay_step)), BN_MOMENTUM_MAX)
    bnm_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=-1)
    
    # checkpoint recovery
    start_epoch = 0
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ Loaded checkpoint from {args.checkpoint_path} (epoch: {start_epoch})")
    
    # training loop
    print(f"\n3. Starting training")
    print(f"   Epochs: {args.epoch}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   LR decay steps: {lr_decay_steps}")
    print(f"   LR decay rates: {lr_decay_rates}")
    
    best_loss = float('inf')
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epoch):
        print(f"\n--- Epoch {epoch+1}/{args.epoch} ---")
        print(f"Current learning rate: {get_current_lr(epoch, args.lr, lr_decay_steps, lr_decay_rates):.6f}")
        print(f"Current BN momentum: {bnm_scheduler.lmbd(bnm_scheduler.last_epoch):.6f}")
        print(f"Time: {datetime.now()}")
        
        # adjust learning rate and BN momentum
        adjust_learning_rate(optimizer, epoch, args.lr, lr_decay_steps, lr_decay_rates)
        bnm_scheduler.step()
        
        # reset numpy random seed
        np.random.seed()
        
        # train
        train_loss, train_trans, train_rot, train_ortho = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, train_writer)
        print(f"Train Loss: {train_loss:.6f} (Trans: {train_trans:.6f}, Rot: {train_rot:.6f}, Ortho: {train_ortho:.6f})")
        
        # validate
        val_loss, val_trans, val_rot, val_ortho = validate(
            model, val_loader, criterion, device, epoch, val_writer)
        print(f"Val Loss: {val_loss:.6f} (Trans: {val_trans:.6f}, Rot: {val_rot:.6f}, Ortho: {val_ortho:.6f})")
        
        # save checkpoint
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint.tar')
        save_dict = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(save_dict, checkpoint_path)
        
        # save best model
        if val_loss < best_loss:
            best_loss = val_loss
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(save_dict, model_path)
            print(f"✅ Saved best model to {model_path}")
    
    # close TensorBoard writers
    train_writer.close()
    val_writer.close()
    
    # training completed
    total_time = time.time() - training_start_time
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Total training time: {format_time(total_time)}")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"TensorBoard logs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

