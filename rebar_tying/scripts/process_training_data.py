#!/usr/bin/env python3
"""
Prepare training data in GraspNet format
- RGB + Depth images
- 6DoF poses
- Segmentation masks
- Camera intrinsics

bash example:
python3 scripts/process_training_data.py --image_dir texture_suppression_model/images/texture_suppressed --depth_dir texture_suppression_model/images/rebar_joint_pose_estimation/Nano0711/Vertical --output_dir datasets --yolo_model texture_suppression_model/runs/pose/train2/weights/best.pt --conf 0.6
"""

import os
import json
import numpy as np
import cv2
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
import re
from sklearn.decomposition import PCA

# Camera intrinsics
FX = 1734.7572357650336
FY = 1734.593101527403
CX = 632.2360387060742
CY = 504.996466076361

CAMERA_INTRINSICS = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])


def depth_to_pointcloud_patch(depth_image, bbox, camera_intrinsics):
    """Extract point cloud from depth image within bbox"""
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = depth_image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    depth_patch = depth_image[y1:y2, x1:x2]
    if depth_patch.size == 0:
        return np.array([]).reshape(0, 3)
    
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    patch_h, patch_w = depth_patch.shape
    u = np.arange(patch_w) + x1
    v = np.arange(patch_h) + y1
    u, v = np.meshgrid(u, v)
    
    z = depth_patch / 1000.0
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid_mask = points[:, 2] > 0
    return points[valid_mask]


def estimate_pose_from_pointcloud(pointcloud):
    """Estimate 6DoF pose from point cloud using PCA"""
    if len(pointcloud) < 3:
        return np.eye(4)
    
    # Center point cloud
    center = np.mean(pointcloud, axis=0)
    centered = pointcloud - center
    
    # PCA for rotation
    pca = PCA(n_components=3)
    pca.fit(centered)
    
    rotation = pca.components_.T
    
    # Ensure right-handed
    rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    rotation[:, 2] /= np.linalg.norm(rotation[:, 2])
    rotation[:, 1] = np.cross(rotation[:, 2], rotation[:, 0])
    rotation[:, 1] /= np.linalg.norm(rotation[:, 1])
    
    # Create 4x4 pose
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = center
    
    return pose


def prepare_scene_data(image_dir, depth_dir, output_dir, yolo_model_path, conf=0.6):
    """
    Prepare data in GraspNet scene format
    """
    print("="*80)
    print("Preparing GraspNet-style Training Data")
    print("="*80)
    
    # Load YOLO model
    print(f"\n1. Loading YOLO model: {yolo_model_path}")
    model = YOLO(yolo_model_path)
    
    # Create output directories
    scenes_dir = os.path.join(output_dir, 'scenes')
    os.makedirs(scenes_dir, exist_ok=True)
    
    rgb_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    print(f"\n2. Found {len(rgb_files)} RGB images")
    
    # Group by scene (by angle or single scene)
    scenes = {}
    
    for idx, rgb_file in enumerate(tqdm(rgb_files, desc="Processing")):
        # Extract angle from filename for grouping
        # Note: labeled angle -> true angle mapping
        angle_map = {'45': 69, '60': 74, '70': 79, '80': 85, '90': 90}
        
        match = re.search(r'p(\d+)_(\d+)_(\d+)_(\d+)', rgb_file)
        if match:
            labeled_angle = match.group(2)
            true_angle = angle_map.get(labeled_angle, labeled_angle)
            scene_key = f"scene_true_angle_{true_angle}"
        else:
            scene_key = "scene_default"
        
        if scene_key not in scenes:
            scene_path = os.path.join(scenes_dir, scene_key)
            os.makedirs(scene_path, exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'label'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'meta'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'pointclouds'), exist_ok=True)  # Add point clouds directory
            scenes[scene_key] = {'path': scene_path, 'frames': 0}
        
        scene_path = scenes[scene_key]['path']
        frame_idx = scenes[scene_key]['frames']
        
        try:
            # Find depth file
            match = re.search(r'p(\d+)_(\d+)_(\d+)_(\d+)', rgb_file)
            if not match:
                continue
            
            pos, angle, dist, rot = match.groups()
            depth_file = None
            for prefix in ['v', 'i']:
                depth_pattern = f"{prefix}_p{pos}_{angle}_{dist}_{rot}_depth_image.tiff"
                potential_path = os.path.join(depth_dir, depth_pattern)
                if os.path.exists(potential_path):
                    depth_file = depth_pattern
                    break
            
            if not depth_file:
                continue
            
            # Load RGB and depth
            rgb_path = os.path.join(image_dir, rgb_file)
            depth_path = os.path.join(depth_dir, depth_file)
            
            rgb_image = cv2.imread(rgb_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if rgb_image is None or depth_image is None:
                continue
            
            # Run YOLO detection
            results = model.predict(rgb_path, conf=conf, verbose=False)
            
            # Create segmentation mask
            h, w = depth_image.shape
            seg_mask = np.zeros((h, w), dtype=np.uint8)
            
            poses_list = []
            cls_indexes = []
            pointclouds_list = []  # Store point clouds for this frame
            
            for det_idx, det in enumerate(results[0].boxes):
                # Get bbox
                x1, y1, x2, y2 = [int(x) for x in det.xyxy[0].cpu().numpy()]
                
                # Fill mask
                cls_id = det_idx + 1
                seg_mask[y1:y2, x1:x2] = cls_id
                
                # Extract point cloud
                bbox = [x1, y1, x2, y2]
                pointcloud = depth_to_pointcloud_patch(depth_image, bbox, CAMERA_INTRINSICS)
                
                if len(pointcloud) < 100:
                    continue
                
                # Estimate pose
                pose = estimate_pose_from_pointcloud(pointcloud)
                poses_list.append(pose)
                cls_indexes.append(cls_id)
                pointclouds_list.append(pointcloud)  # Store point cloud
            
            if len(poses_list) == 0:
                continue
            
            # Save RGB and depth
            rgb_out = os.path.join(scene_path, 'rgb', f"{frame_idx:04d}.png")
            depth_out = os.path.join(scene_path, 'depth', f"{frame_idx:04d}.png")
            label_out = os.path.join(scene_path, 'label', f"{frame_idx:04d}.png")
            
            cv2.imwrite(rgb_out, rgb_image)
            cv2.imwrite(depth_out, depth_image)
            cv2.imwrite(label_out, seg_mask)
            
            # Save meta (GraspNet format)
            import scipy.io as sio
            meta = {
                'cls_indexes': np.array(cls_indexes),
                'poses': np.array(poses_list).transpose(0, 1, 2),  # Nx4x4
                'intrinsic_matrix': CAMERA_INTRINSICS,
                'factor_depth': 1000.0  # mm to m
            }
            meta_out = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
            sio.savemat(meta_out, meta)
            
            # Save point clouds for each object
            for obj_idx, pointcloud in enumerate(pointclouds_list):
                pc_out = os.path.join(scene_path, 'pointclouds', f"{frame_idx:04d}_obj{obj_idx}.npy")
                np.save(pc_out, pointcloud)
            
            scenes[scene_key]['frames'] += 1
        
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Save camera intrinsics for each scene
    for scene_key, scene_info in scenes.items():
        camK_out = os.path.join(scene_info['path'], 'camK.npy')
        np.save(camK_out, CAMERA_INTRINSICS)
    
    print(f"\n✓ Created {len(scenes)} scenes")
    print(f"  Output directory: {output_dir}")
    for scene_key, scene_info in scenes.items():
        print(f"  {scene_key}: {scene_info['frames']} frames")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GraspNet-style training data")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--depth_dir", required=True)
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--yolo_model", default="texture_suppression_model/runs/pose/train2/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    
    args = parser.parse_args()
    
    prepare_scene_data(
        args.image_dir,
        args.depth_dir,
        args.output_dir,
        args.yolo_model,
        args.conf
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

