#!/usr/bin/env python3
"""
# Prepare training data in GraspNet format: RGB + Depth images + 6DoF poses + Segmentation masks + Camera intrinsics

# bash example for rebar_tying dataset:

python3 rebar_tying/scripts/process_training_data.py \
    --depth_dir rebar_tying/texture_suppression_model/images/rebar_joint_pose_estimation/Nano0711 \
    --output_dir rebar_tying/datasets \
    --yolo_model rebar_tying/texture_suppression_model/runs/pose/train2/weights/best.pt \
    --conf 0.7 --imgsz 960 --device cuda:0 --min_points 200 --roi_margin 100
    
    Note: 
    - Both RGB and Depth files are searched in depth_dir/Vertical/ or depth_dir/Incline/ subdirectories
    - RGB: {v|i}_p{pos}_{angle}_{dist}_{rot}_depth_filtered_image.jpg
    - Depth: {v|i}_p{pos}_{angle}_{dist}_{rot}_depth_image.tiff
    - Vertical subdirectory: only matches v_* files
    - Incline subdirectory: only matches i_* files

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
from sklearn.cluster import DBSCAN
import collections
import open3d as o3d

# Camera intrinsics 
FX = 1734.7572357650336
FY = 1734.593101527403
CX = 632.2360387060742
CY = 504.996466076361

# Camera intrinsics for rebar_tying dataset
CAMERA_INTRINSICS = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
])

# -----------------------------generate local point cloud------------------------------
def depth_to_pointcloud_patch(depth_image, rgb_image, bbox, camera_intrinsics, window_size=200):
    """
    Extract point cloud from depth image within a fixed-size window centered at bbox center.
    Similar to cylinder_fitting.py's approach of focusing on a 200x200 window around the keypoint.
    
    Args:
        bbox: bounding box [x1, y1, x2, y2]
        window_size: size of the square window (default 200x200 like cylinder_fitting.py)
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = depth_image.shape[:2]
    
    # Calculate bbox center (keypoint)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Create fixed-size window centered at bbox center (like cylinder_fitting.py's half_win=100)
    half_win = window_size // 2
    x_min = max(0, cx - half_win)
    x_max = min(w, cx + half_win)
    y_min = max(0, cy - half_win)
    y_max = min(h, cy + half_win)
    
    # crop depth image
    # note: depth_image should be in millimeters, here convert to meters
    depth_crop = depth_image[y_min:y_max, x_min:x_max].astype(np.float32) / 1000.0
    
    # crop RGB image
    if rgb_image is not None:
        hrgb, wrgb = rgb_image.shape[:2]
        if (hrgb != h) or (wrgb != w):
            # RGB and depth image size are not consistent, need to resize RGB to depth image size
            rgb_resized = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_LINEAR)
            # convert to RGB format
            rgb_resized = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB)
            rgb_crop = rgb_resized[y_min:y_max, x_min:x_max]
        else:
            rgb_crop = cv2.cvtColor(rgb_image[y_min:y_max, x_min:x_max], cv2.COLOR_BGR2RGB)
    else:
        rgb_crop = None
    
    if depth_crop.size == 0:
        return np.array([]).reshape(0, 3)
    
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # use meshgrid to generate pixel coordinates (use the window coordinates)
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    
    # z is already in meters, directly use
    z = depth_crop
    x3d = (xx - cx) * z / fx
    y3d = (yy - cy) * z / fy
    
    # filter valid depth
    valid = (z > 0) & (z < 0.5) & np.isfinite(z) & np.isfinite(x3d) & np.isfinite(y3d)
    
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    
    # stack xyz coordinates
    xyz = np.stack((x3d[valid], y3d[valid], z[valid]), axis=-1)
    
    return xyz

# -----------------------------point cloud downsampling function------------------------------
def downsample_points(points_np, voxel_size=0.005):
    """Downsample point cloud using voxel grid"""
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points_np)
    pcd_down = pcd_tmp.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_down.points)

# -----------------------------cylinder fitting function (least squares)------------------------------
def fit_cylinder_least_squares(points):
    """Fit cylinder using least squares method"""
    if len(points) < 10:
        return None
    
    pca = PCA(n_components=3)
    pca.fit(points)
    axis_dir = pca.components_[0]

    projections = points @ axis_dir
    h_min, h_max = projections.min(), projections.max()

    points_proj = points - np.outer(projections, axis_dir)
    plane_x = pca.components_[1]
    plane_y = pca.components_[2]

    u = points_proj @ plane_x
    v = points_proj @ plane_y

    A = np.column_stack((2*u, 2*v, np.ones_like(u)))
    b = u**2 + v**2

    x = np.linalg.lstsq(A, b, rcond=None)[0]
    u0, v0, c = x
    radius = np.sqrt(c + u0**2 + v0**2)

    axis_point = u0 * plane_x + v0 * plane_y + axis_dir * h_min

    return axis_point, axis_dir, radius, h_min, h_max

# -----------------------------canonicalize coordinate system------------------------------
def canonicalize_axes_opencv(z_axis, x_hint, ref_z=np.array([0.,0.,1.], dtype=np.float32)):
    """
    Canonicalize local coordinate system to OpenCV camera frame:
      1) z aligns with ref_z (camera +Z)
      2) x is the orthogonal projection of x_hint onto z's plane
      3) Right-handed system, with y=z×x also aligned with ref_z
    Input:
      z_axis: main axis direction (e.g. cyl1_dir)
      x_hint: secondary axis direction (e.g. cyl2_dir)
    Returns:
      R: 3x3 rotation matrix (column vectors are x,y,z)
    """
    z = z_axis.astype(np.float32)
    z /= (np.linalg.norm(z) + 1e-9)

    # (1) Force z to align with ref_z
    if float(np.dot(z, ref_z)) < 0.0:
        z = -z

    # (2) Use x_hint to construct x on z's orthogonal plane
    xh = x_hint.astype(np.float32)
    xh /= (np.linalg.norm(xh) + 1e-9)
    x = xh - float(np.dot(xh, z)) * z
    if np.linalg.norm(x) < 1e-9:  # Degenerate case: x_hint parallel to z, pick arbitrary orthogonal axis
        x = np.array([1.,0.,0.], dtype=np.float32) - float(np.dot([1.,0.,0.], z)) * z
    x /= (np.linalg.norm(x) + 1e-9)

    # (3) Right-handed coordinate system
    y = np.cross(z, x)
    y /= (np.linalg.norm(y) + 1e-9)

    # Optional: ensure y aligns with ref_z (unique orientation)
    if float(np.dot(y, ref_z)) < 0.0:
        x = -x
        y = -y

    R = np.stack([x, y, z], axis=1)  # columns are x,y,z
    # Numerical robustness: ensure det>0
    if np.linalg.det(R) < 0:
        x = -x
        R = np.stack([x, y, z], axis=1)
    return R

# -----------------------------calculate line intersection------------------------------
def calculate_line_intersection(line1_point, line1_dir, line2_point, line2_dir):
    """Calculate intersection point of two lines in 3D"""
    # Line 1: P1 + t1 * D1
    # Line 2: P2 + t2 * D2
    # Solve: P1 + t1 * D1 = P2 + t2 * D2
    
    P1, D1 = line1_point, line1_dir
    P2, D2 = line2_point, line2_dir
    
    # Cross product of directions
    cross_D1_D2 = np.cross(D1, D2)
    cross_norm = np.linalg.norm(cross_D1_D2)
    
    if cross_norm < 1e-6:  # Lines are parallel
        # Return midpoint of closest points
        t1 = np.dot(P2 - P1, D1) / np.dot(D1, D1)
        closest_point = P1 + t1 * D1
        return closest_point
    
    # Calculate closest points on both lines
    w0 = P1 - P2
    a = np.dot(D1, D1)
    b = np.dot(D1, D2)
    c = np.dot(D2, D2)
    d = np.dot(D1, w0)
    e = np.dot(D2, w0)
    
    denom = a * c - b * b
    if abs(denom) < 1e-6:
        # Lines are parallel, return midpoint
        return (P1 + P2) / 2
    
    t1 = (b * e - c * d) / denom
    t2 = (a * e - b * d) / denom
    
    point1 = P1 + t1 * D1
    point2 = P2 + t2 * D2
    
    # Return midpoint of closest points
    intersection = (point1 + point2) / 2
    return intersection

# -----------------------------estimate pose from point cloud------------------------------
def estimate_pose_from_pointcloud(pointcloud, stats=None):
    """
    Estimate 6DoF pose from point cloud using cylinder fitting.
    
    This implementation follows the same clustering strategy:
    - Uses DBSCAN with min_samples=30 to detect clusters
    - Extracts top 3 clusters
    - Merges 2nd and 3rd largest clusters to represent one rebar
    - Fits cylinders to find intersection point and orientation
    """
    if len(pointcloud) < 50:
        return np.eye(4)
    
    try:
        # 1. build point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        # 2. noise removal
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd_clean.points) < 30:
            return np.eye(4)
        
        points = np.asarray(pcd_clean.points)
        
        # 3. Limit points for DBSCAN to speed up clustering
        MAX_PTS = 5000
        if len(points) > MAX_PTS:
            rng = np.random.default_rng()
            sel = rng.choice(len(points), size=MAX_PTS, replace=False)
            points = points[sel]
        
        # Lightweight whitening (faster than StandardScaler)
        mean = points.mean(axis=0, keepdims=True)
        std = points.std(axis=0, keepdims=True) + 1e-9
        points_scaled = (points - mean) / std
        
        # DBSCAN with parallel processing
        db = DBSCAN(eps=0.3, min_samples=30, n_jobs=-1).fit(points_scaled)
        labels = db.labels_
        
        # 4. extract the main clusters (top 3, then merge 2nd and 3rd)
        valid_labels = labels[labels != -1]
        unique_labels = set(valid_labels)
        if len(unique_labels) < 3:  # need at least 3 clusters for merging
            # fallback: try with top 2 clusters if only 2 available
            if stats is not None:
                if len(unique_labels) >= 2:
                    stats['top2_fallback'] += 1
                else:
                    stats['cluster_too_few'] += 1
            if len(unique_labels) >= 2:
                top2_labels = [label for label, _ in collections.Counter(labels[labels != -1]).most_common(2)]
                if len(top2_labels) == 2:
                    cyl1_points = points[labels == top2_labels[0]]
                    cyl2_points = points[labels == top2_labels[1]]
                    if len(cyl1_points) >= 10 and len(cyl2_points) >= 10:
                        cyl1_points_ds = downsample_points(cyl1_points, voxel_size=0.001)
                        cyl2_points_ds = downsample_points(cyl2_points, voxel_size=0.001)
                        if len(cyl1_points_ds) >= 5 and len(cyl2_points_ds) >= 5:
                            cyl1_params = fit_cylinder_least_squares(cyl1_points_ds)
                            cyl2_params = fit_cylinder_least_squares(cyl2_points_ds)
                            if cyl1_params is not None and cyl2_params is not None:
                                cyl1_point, cyl1_dir, cyl1_radius, cyl1_hmin, cyl1_hmax = cyl1_params
                                cyl2_point, cyl2_dir, cyl2_radius, cyl2_hmin, cyl2_hmax = cyl2_params
                                intersection_point = calculate_line_intersection(
                                    cyl1_point, cyl1_dir, cyl2_point, cyl2_dir
                                )
                                # Canonicalize coordinate system: z aligns with camera +Z
                                R = canonicalize_axes_opencv(z_axis=cyl1_dir, x_hint=cyl2_dir)
                                pose_matrix = np.eye(4, dtype=np.float32)
                                pose_matrix[:3, :3] = R
                                pose_matrix[:3, 3] = intersection_point.astype(np.float32)
                                return pose_matrix
            return np.eye(4)
        
        label_counts = collections.Counter(labels[labels != -1])
        top3_labels = [label for label, _ in label_counts.most_common(3)]
        
        if len(top3_labels) < 3:
            return np.eye(4)
        
        cyl1_points = points[labels == top3_labels[0]]  # main cluster
        cyl2_points_1 = points[labels == top3_labels[1]]  # 2nd cluster
        cyl2_points_2 = points[labels == top3_labels[2]]  # 3rd cluster
        
        # 5. downsampling (downsample each cluster BEFORE merging)
        cyl1_points_ds = downsample_points(cyl1_points, voxel_size=0.001)
        cyl2_points_1_ds = downsample_points(cyl2_points_1, voxel_size=0.001)
        cyl2_points_2_ds = downsample_points(cyl2_points_2, voxel_size=0.001)
        
        # merge 2nd and 3rd clusters
        cyl2_points_ds = np.vstack((cyl2_points_1_ds, cyl2_points_2_ds))
        
        if len(cyl1_points_ds) < 5 or len(cyl2_points_ds) < 5:
            return np.eye(4)
        
        # 6. fit two cylinders
        cyl1_params = fit_cylinder_least_squares(cyl1_points_ds)
        cyl2_params = fit_cylinder_least_squares(cyl2_points_ds)
        
        if cyl1_params is None or cyl2_params is None:
            return np.eye(4)
        
        cyl1_point, cyl1_dir, cyl1_radius, cyl1_hmin, cyl1_hmax = cyl1_params
        cyl2_point, cyl2_dir, cyl2_radius, cyl2_hmin, cyl2_hmax = cyl2_params
        
        # 7. calculate the intersection point of the two cylinder axes
        intersection_point = calculate_line_intersection(
            cyl1_point, cyl1_dir, cyl2_point, cyl2_dir
        )
        
        # 8. canonicalize coordinate system: z aligns with camera +Z
        R = canonicalize_axes_opencv(z_axis=cyl1_dir, x_hint=cyl2_dir)
        pose_matrix = np.eye(4, dtype=np.float32)
        pose_matrix[:3, :3] = R
        pose_matrix[:3, 3] = intersection_point.astype(np.float32)
        
        # Mark successful top3 clustering
        if stats is not None:
            stats['top3_success'] += 1
        
        return pose_matrix
        
    except Exception as e:
        print(f"Cylinder fitting failed: {e}")
        # fallback to simple centroid method
        center = np.mean(pointcloud, axis=0)
        pose = np.eye(4)
        pose[:3, 3] = center
        return pose

# -----------------------------prepare scene data------------------------------
def prepare_scene_data(depth_dir, output_dir, yolo_model_path, conf=0.6, imgsz=640, device=None, min_points=100, roi_margin=50):
    """
    Prepare data in GraspNet scene format
    
    Args:
        roi_margin: pixel margin from image borders to exclude from ROI (default 50)
                    Only bbox centers within ROI are kept (ignores edge detections)
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
    
    # find all RGB files in depth_dir (depth_filtered_image.jpg)
    # support Vertical and Incline subdirectory
    # Vertical directory only matches v_ files, Incline directory only matches i_ files
    '''
    # example directory structure
    depth_dir/
    ├── Vertical/
    │   ├── v_*_depth_filtered_image.jpg  ← RGB
    │   └── v_*_depth_image.tiff            ← Depth
    └── Incline/
        ├── i_*_depth_filtered_image.jpg    ← RGB
        └── i_*_depth_image.tiff            ← Depth

    '''
    rgb_files = []
    subdirs = ['Vertical', 'Incline', '']  # support subdirectory or directly in depth_dir
    for subdir in subdirs:
        search_dir = os.path.join(depth_dir, subdir) if subdir else depth_dir
        if not os.path.exists(search_dir):
            continue
        # determine file prefix based on subdirectory
        expected_prefix = None
        if subdir == 'Vertical':
            expected_prefix = 'v'
        elif subdir == 'Incline':
            expected_prefix = 'i'
        
        for f in os.listdir(search_dir):
            if not f.endswith('_depth_filtered_image.jpg'):
                continue
            # 验证文件前缀是否匹配子目录
            if expected_prefix and not f.startswith(f'{expected_prefix}_'):
                continue
            rgb_files.append((subdir, f))
    
    rgb_files.sort()
    print(f"\n2. Found {len(rgb_files)} RGB images (depth_filtered_image.jpg)")

    # Group by scene (by subdirectory: Vertical or Incline)
    scenes = {}
    
    # Debug counters
    stats = {
        'total_frames': 0,
        'no_detections': 0,
        'detections_filtered': 0,
        'fitting_failed': 0,
        'success_frames': 0,
        'low_points': 0,
        'identity_pose': 0,
        'cluster_too_few': 0,
        'top2_fallback': 0,
        'top3_success': 0
    }
    
    for idx, (subdir, rgb_file) in enumerate(tqdm(rgb_files, desc="Processing")):
        # group by scene (by subdirectory: Vertical or Incline)
        if subdir:
            # use subdirectory name as scene name (convert to lowercase)
            scene_key = f"scene_{subdir.lower()}"
        else:
            # if no subdirectory, use default name
            scene_key = "scene_default"
        
        if scene_key not in scenes:
            scene_path = os.path.join(scenes_dir, scene_key)
            os.makedirs(scene_path, exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'rgb'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'depth'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'label'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'meta'), exist_ok=True)
            os.makedirs(os.path.join(scene_path, 'pointclouds'), exist_ok=True)
            scenes[scene_key] = {'path': scene_path, 'frames': 0, 'Kd': None}
        
        scene_path = scenes[scene_key]['path']
        frame_idx = scenes[scene_key]['frames']
        
        try:
            stats['total_frames'] += 1
            # extract information from RGB file name and find corresponding depth file
            match = re.search(r'(v|i)_p(\d+)_(\d+)_(\d+)_(\d+)_depth_filtered_image\.jpg', rgb_file)
            if not match:
                continue
            
            prefix, pos, angle, dist, rot = match.groups()
            
            # verify prefix matches subdirectory
            if subdir == 'Vertical' and prefix != 'v':
                print(f"   Warning: Vertical subdir but file prefix is '{prefix}': {rgb_file}")
                continue
            elif subdir == 'Incline' and prefix != 'i':
                print(f"   Warning: Incline subdir but file prefix is '{prefix}': {rgb_file}")
                continue
            
            # build RGB and depth file paths (in the same subdirectory)
            if subdir:
                base_dir = os.path.join(depth_dir, subdir)
            else:
                base_dir = depth_dir
            
            rgb_path = os.path.join(base_dir, rgb_file)
            depth_file = f"{prefix}_p{pos}_{angle}_{dist}_{rot}_depth_image.tiff"
            depth_path = os.path.join(base_dir, depth_file)
            
            # check if files exist
            if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
                continue
            
            rgb_image = cv2.imread(rgb_path)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            if rgb_image is None or depth_image is None:
                continue
            
            # Run YOLO detection on RGB image
            results = model.predict(rgb_path, conf=conf, imgsz=imgsz, device=device, verbose=False)
            
            # Create segmentation mask
            h, w = depth_image.shape
            seg_mask = np.zeros((h, w), dtype=np.uint8)
            
            poses_list = []
            cls_indexes = []
            pointclouds_list = []  # Store point clouds for this frame
            
            # size matching: YOLO generated bbox on RGB image, need to convert to depth image size
            hrgb, wrgb = rgb_image.shape[:2]
            scale_x = float(w) / float(wrgb) if wrgb > 0 else 1.0
            scale_y = float(h) / float(hrgb) if hrgb > 0 else 1.0
            # derive depth-sized intrinsics from given K (assumed RGB-sized originally)
            Kd = CAMERA_INTRINSICS.copy().astype(np.float32)
            Kd[0, 0] *= scale_x
            Kd[1, 1] *= scale_y
            Kd[0, 2] *= scale_x
            Kd[1, 2] *= scale_y
            # record scene-level Kd (consistent with the depth size of the scene)
            if scenes[scene_key]['Kd'] is None:
                scenes[scene_key]['Kd'] = Kd

            kept = []  # final objects kept for writing labels
            boxes = getattr(results[0], "boxes", None) 
            total_detections = 0 if boxes is None else len(boxes)
            if not boxes or total_detections == 0:
                stats['no_detections'] += 1
                continue

            for det_idx, det in enumerate(results[0].boxes):
                # confidence filter
                if hasattr(det, 'conf') and float(det.conf.cpu().numpy().reshape(-1)[0]) < conf:
                    continue
                # Get bbox (xyxy on RGB), then map to depth size
                x1r, y1r, x2r, y2r = [float(x) for x in det.xyxy[0].cpu().numpy()]
                x1 = int(round(x1r * scale_x)); x2 = int(round(x2r * scale_x))
                y1 = int(round(y1r * scale_y)); y2 = int(round(y2r * scale_y))
                # shrink bbox margins to reduce background
                bw = max(1, x2 - x1); bh = max(1, y2 - y1)   # width and height of the bbox on the RGB image
                cx = (x1 + x2) * 0.5; cy = (y1 + y2) * 0.5   # center of the bbox on the RGB image
                
                # ROI filter: only keep bbox centers within valid ROI (exclude image edges)
                if cx < roi_margin or cx > w - roi_margin or cy < roi_margin or cy > h - roi_margin:
                    continue
                
                shrink = 0.08
                bw2 = max(6.0, bw * (1.0 - shrink)); bh2 = max(6.0, bh * (1.0 - shrink))
                x1 = int(round(cx - bw2 * 0.5)); x2 = int(round(cx + bw2 * 0.5))
                y1 = int(round(cy - bh2 * 0.5)); y2 = int(round(cy + bh2 * 0.5))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                
                # Fill mask: fill the bbox on the segmentation mask with the class id
                raw_cls = int(det.cls.cpu().item()) if hasattr(det, 'cls') else 0
                cls_id = raw_cls + 1   # keep 1..N semantics (0 for background/unlabeled)
                seg_mask[y1:y2, x1:x2] = cls_id
                
                # Extract point cloud (using RGB-D alignment method)
                bbox = [x1, y1, x2, y2]
                pointcloud_raw = depth_to_pointcloud_patch(depth_image, rgb_image, bbox, Kd)
                
                if len(pointcloud_raw) < min_points:
                    stats['low_points'] += 1
                    continue
                
                # NO pre-downsampling
                pose = estimate_pose_from_pointcloud(pointcloud_raw, stats=stats)
                # skip invalid (identity) pose
                if np.allclose(pose, np.eye(4), atol=1e-4):
                    stats['identity_pose'] += 1
                    continue
                poses_list.append(pose)
                cls_indexes.append(cls_id)
                # Store original point cloud (not downsampled) for training - more points
                pointclouds_list.append(pointcloud_raw)
                conf_val = float(det.conf.cpu().numpy().reshape(-1)[0]) if hasattr(det, 'conf') else 1.0
                kept.append({'cls_id': cls_id, 'bbox': [x1, y1, x2, y2], 'conf': conf_val})
            
            # Debug: print why frames are skipped
            if len(poses_list) == 0:
                if total_detections == 0:
                    stats['no_detections'] += 1
                elif len(kept) == 0:
                    stats['detections_filtered'] += 1
                else:
                    stats['fitting_failed'] += 1
                continue
            
            stats['success_frames'] += 1
            
            # Save RGB and depth (ensure depth is uint16 to avoid CV_8U fallback)
            # Use fast PNG compression for speed
            png_fast = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # 0-9, 1 is fast with good quality
            
            rgb_out = os.path.join(scene_path, 'rgb', f"{frame_idx:04d}.png")
            depth_out = os.path.join(scene_path, 'depth', f"{frame_idx:04d}.png")
            label_out = os.path.join(scene_path, 'label', f"{frame_idx:04d}.png")
            
            cv2.imwrite(rgb_out, rgb_image, png_fast)
            # convert depth to uint16 in millimeters if needed
            d = depth_image
            if d.dtype != np.uint16:
                d_mm = d.astype(np.float32)
                # heuristic: if values look like meters, convert to mm
                vmax = float(np.nanmax(d_mm)) if np.isfinite(d_mm).any() else 0.0
                if vmax <= 50.0:  # very likely meters
                    d_mm = d_mm * 1000.0
                d_mm = np.clip(d_mm, 0, 65535).astype(np.uint16)
                depth_to_save = d_mm
            else:
                depth_to_save = d
            cv2.imwrite(depth_out, depth_to_save, png_fast)
            cv2.imwrite(label_out, seg_mask, png_fast)
            
            # Save meta (GraspNet format) without compression for speed
            import scipy.io as sio
            meta = {
                'cls_indexes': np.array(cls_indexes, dtype=np.int32),
                'poses': np.asarray(poses_list, dtype=np.float32),  # (N,4,4)
                'intrinsic_matrix': Kd.astype(np.float32),
                'factor_depth': np.float32(1000.0)  # mm to m
            }
            meta_out = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
            sio.savemat(meta_out, meta, do_compression=False)
            
            # Save point clouds for each object (.npy only - PLY disabled for speed)
            SAVE_PLY = False  # PLY files not needed for training
            for obj_idx, pointcloud in enumerate(pointclouds_list):
                pc_out = os.path.join(scene_path, 'pointclouds', f"{frame_idx:04d}_obj{obj_idx}.npy")
                np.save(pc_out, pointcloud.astype(np.float32))
                if SAVE_PLY:
                    # PLY (binary) via Open3D
                    try:
                        pcd_o3d = o3d.geometry.PointCloud()
                        pcd_o3d.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float64))
                        pc_out_ply = os.path.join(scene_path, 'pointclouds', f"{frame_idx:04d}_obj{obj_idx}.ply")
                        o3d.io.write_point_cloud(pc_out_ply, pcd_o3d, write_ascii=False)
                    except Exception as e:
                        print(f"   Warning: failed to write PLY for {frame_idx:04d}_obj{obj_idx}: {e}")
            # Save depth-sized xyxy labels with conf for reproducibility
            labels_dir = os.path.join(scene_path, 'labels_xyxy_depth')
            os.makedirs(labels_dir, exist_ok=True)
            label_path = os.path.join(labels_dir, f"{frame_idx:04d}.txt")
            # write kept detections with true cls_id and conf
            with open(label_path, 'w') as f:
                for k in kept:
                    x1d, y1d, x2d, y2d = k['bbox']
                    f.write(f"{k['cls_id']} {x1d} {y1d} {x2d} {y2d} {k['conf']:.4f}\n")
            
            scenes[scene_key]['frames'] += 1
        
        except Exception as e:
            print(f"   Error: {e}")
            continue
    
    # Save camera intrinsics for each scene (use depth-sized Kd)
    for scene_key, scene_info in scenes.items():
        camK_out = os.path.join(scene_info['path'], 'camK.npy')
        Kd = scene_info.get('Kd', CAMERA_INTRINSICS).astype(np.float32)
        np.save(camK_out, Kd)
    
    print(f"\n✓ Created {len(scenes)} scenes")
    print(f"  Output directory: {output_dir}")
    for scene_key, scene_info in scenes.items():
        print(f"  {scene_key}: {scene_info['frames']} frames")
    
    # print(f"\nDebug Statistics:")
    # print(f"  Total frames processed: {stats['total_frames']}")
    # print(f"  Successful frames: {stats['success_frames']}")
    # print(f"  Frames with no YOLO detections: {stats['no_detections']}")
    # print(f"  Frames with detections filtered out: {stats['detections_filtered']}")
    # print(f"  Frames with cylinder fitting failed: {stats['fitting_failed']}")
    # print(f"  Detections with low point counts (<{min_points}): {stats['low_points']}")
    # print(f"  Detections with identity pose: {stats['identity_pose']}")
    # print(f"\nClustering Statistics:")
    # print(f"  Top3 clustering success: {stats['top3_success']}")
    # print(f"  Top2 fallback: {stats['top2_fallback']}")
    # print(f"  Cluster too few (<2): {stats['cluster_too_few']}")

# -----------------------------main function------------------------------
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare GraspNet-style training data")
    parser.add_argument(
        "--depth_dir", 
        required=True,
        help="Directory containing both RGB and Depth images in Vertical/ and Incline/ subdirectories"
    )
    parser.add_argument("--output_dir", default="datasets")
    parser.add_argument("--yolo_model", default="texture_suppression_model/runs/pose/train2/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min_points", type=int, default=100)
    parser.add_argument("--roi_margin", type=int, default=50, 
                        help="Pixel margin from image borders to exclude from ROI (default 50)")
    
    args = parser.parse_args()
    
    prepare_scene_data(
        args.depth_dir,
        args.output_dir,
        args.yolo_model,
        args.conf,
        args.imgsz,
        args.device,
        args.min_points,
        args.roi_margin
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()