#!/usr/bin/env python3
"""
可视化YOLO检测的bbox中心点(关键点)和估计的法向线
"""
import os
import sys
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO
import scipy.io as sio
import glob

# Camera intrinsics
FX = 1734.7572357650336
FY = 1734.593101527403
CX = 632.2360387060742
CY = 504.996466076361

# -----------------------------frame converters (only for visualization)------------------------------
def pose_cv_to_viz(pose_cv: np.ndarray, mode: str = "open3d") -> np.ndarray:
    """
    Convert a pose defined in OpenCV camera frame (x right, y down, z forward)
    to a visualization frame.

    mode:
      - "open3d": x right, y up, z forward  -> flip Y
      - "opengl": x right, y up, z backward -> flip Y and Z
      - "opencv": no change
    """
    T = np.eye(4, dtype=np.float32)
    if mode == "open3d":
        T[:3, :3] = np.diag([1, -1, 1]).astype(np.float32)
    elif mode == "opengl":
        T[:3, :3] = np.diag([1, -1, -1]).astype(np.float32)
    else:  # "opencv"
        return pose_cv.astype(np.float32)
    return (T @ pose_cv).astype(np.float32)

# -----------------------------depth to point cloud function------------------------------
def depth_to_pointcloud_patch(depth_image, rgb_image, bbox, camera_intrinsics, window_size=200, mode="center_win"):
    """
    Extract point cloud from depth image.
    
    Args:
        bbox: bounding box [x1, y1, x2, y2]
        camera_intrinsics: camera intrinsics matrix
        window_size: size of the square window when mode="center_win" (default 200x200)
        mode: "center_win" = 200x200 window around center, "bbox" = use entire bbox
    """
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    h, w = depth_image.shape[:2]
    
    if mode == "center_win" and window_size is not None:
        # Calculate bbox center (keypoint) and create fixed-size window
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half_win = window_size // 2
        x_min = max(0, cx - half_win)
        x_max = min(w, cx + half_win)
        y_min = max(0, cy - half_win)
        y_max = min(h, cy + half_win)
    else:
        # Use entire bbox
        x_min = max(0, x1)
        x_max = min(w, x2)
        y_min = max(0, y1)
        y_max = min(h, y2)
    
    # crop depth image (in millimeters, convert to meters)
    depth_crop = depth_image[y_min:y_max, x_min:x_max].astype(np.float32) / 1000.0
    
    if depth_crop.size == 0:
        return np.array([]).reshape(0, 3)
    
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    # use meshgrid to generate pixel coordinates
    xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
    
    # z is already in meters
    z = depth_crop
    x3d = (xx - cx) * z / fx
    y3d = (yy - cy) * z / fy
    
    # Adaptive depth filtering (replace fixed 0.5m threshold)
    z_valid = np.isfinite(z) & (z > 0) & np.isfinite(x3d) & np.isfinite(y3d)
    if np.any(z_valid):
        z_med = np.median(z[z_valid])
        # Keep points within ±8cm of median (adjust 0.08 to 0.04-0.10 as needed)
        valid = z_valid & (np.abs(z - z_med) < 0.08)
    else:
        valid = z_valid
    
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    
    # stack xyz coordinates
    xyz = np.stack((x3d[valid], y3d[valid], z[valid]), axis=-1)
    
    return xyz

# -----------------------------project 3D points to 2D pixels function------------------------------
def project_3d_to_2d(points_3d, camera_intrinsics):
    """Project 3D points to 2D pixels"""
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    u = fx * (x / z) + cx
    v = fy * (y / z) + cy
    
    return np.stack([u, v], axis=1)

# -----------------------------visualize keypoint (bbox center) and estimated normal from pose function------------------------------
def visualize_keypoint_with_normal(dataset_root, scene_name="scene_incline", frame_idx=0, obj_idx=0):
    """Visualize keypoint (bbox center) and estimated normal from pose"""
    
    scene_path = os.path.join(dataset_root, 'scenes', scene_name)
    if not os.path.exists(scene_path):
        print(f"Scene not found: {scene_path}")
        return
    
    # Load meta to get pose
    meta_file = os.path.join(scene_path, 'meta', f"{frame_idx:04d}.mat")
    if not os.path.exists(meta_file):
        print(f"Meta file not found: {meta_file}")
        return
    
    # load meta file
    meta = sio.loadmat(meta_file)
    poses = meta['poses']  # (N, 4, 4)
    intrinsic_matrix = meta['intrinsic_matrix']
    
    if obj_idx >= len(poses):
        print(f"Object index {obj_idx} >= {len(poses)} poses available")
        return
    
    pose = poses[obj_idx]
    
    # Load point cloud
    pc_file = os.path.join(scene_path, 'pointclouds', f"{frame_idx:04d}_obj{obj_idx}.npy")
    if not os.path.exists(pc_file):
        print(f"Point cloud not found: {pc_file}")
        return
    
    pointcloud = np.load(pc_file)
    
    # Load RGB for background
    rgb_file = os.path.join(scene_path, 'rgb', f"{frame_idx:04d}.png")
    rgb_image = cv2.imread(rgb_file) if os.path.exists(rgb_file) else None
    print(f"\nVisualizing: {scene_name}, Frame {frame_idx}, Object {obj_idx}")
    print(f"Point cloud: {len(pointcloud)} points")
    print(f"Pose:\n{pose}")
    
    # Extract keypoint (intersection/origin from pose)
    keypoint_3d = pose[:3, 3]
    print(f"Keypoint 3D: {keypoint_3d}")
    
    # Extract normal (Z-axis from pose)
    normal_3d = pose[:3, 2]
    print(f"Normal (Z-axis): {normal_3d}")
    print(f"Normal magnitude: {np.linalg.norm(normal_3d):.4f}")
    
    # Pose is already in world coordinate system (Z=up), no transformation needed
    pose_viz = pose.astype(np.float64)
    keypoint_3d_viz = pose_viz[:3, 3]
    normal_3d_viz = pose_viz[:3, 2]
    
    # Build visualization
    geometries = []
    
    # 1. Point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geometries.append(pcd)
    
    # 2. Keypoint (red sphere) - transformed
    keypoint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    keypoint_sphere.translate(keypoint_3d_viz)
    keypoint_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    geometries.append(keypoint_sphere)
    
    # 3. Normal line (blue line extending from keypoint) - transformed
    normal_length = 0.1  # 10cm
    normal_end = keypoint_3d_viz + normal_3d_viz * normal_length
    
    normal_line = o3d.geometry.LineSet()
    normal_line.points = o3d.utility.Vector3dVector([keypoint_3d_viz, normal_end])
    normal_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    normal_line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]])  # Blue
    geometries.append(normal_line)
    
    # 4. Coordinate frame - transformed
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    coord_frame.transform(pose_viz)
    geometries.append(coord_frame)
    
    # 5. If RGB available, create textured background
    if rgb_image is not None:
        # Project keypoint to 2D
        keypoint_2d = project_3d_to_2d(keypoint_3d.reshape(1, 3), intrinsic_matrix)[0]
        print(f"Keypoint 2D: ({keypoint_2d[0]:.1f}, {keypoint_2d[1]:.1f})")
    
    # Note: RGB overlay saving is now handled in browse_dataset's save_rgb_overlay function
    # This function (visualize_keypoint_with_normal) only does 3D visualization
    
    # Visualize
    print("\nOpen3D visualization:")
    print("  - Red sphere: Keypoint (bbox center in 3D)")
    print("  - Blue line: Normal vector (Z-axis)")
    print("  - XYZ frame: Pose coordinate system")
    print("  - Gray points: Point cloud")
    
    # Use VisualizerWithKeyCallback to set camera view
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=f"Keypoint: {scene_name} Frame {frame_idx} Obj {obj_idx}", width=1024, height=768)
    for geom in geometries:
        vis.add_geometry(geom)
    vis.get_render_option().background_color = np.array([1,1,1])
    vis.get_view_control().set_front([1,0,0])  # Camera looks from X direction
    vis.get_view_control().set_up([0,0,1])     # Z-up in world
    vis.run()
    vis.destroy_window()
    
    # Also print 2D bbox info if available
    labels_file = os.path.join(scene_path, 'labels_xyxy_depth', f"{frame_idx:04d}.txt")
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            lines = f.readlines()
            if obj_idx < len(lines):
                parts = lines[obj_idx].strip().split()
                cls_id, x1, y1, x2, y2, conf = parts
                cx_2d = (float(x1) + float(x2)) / 2
                cy_2d = (float(y1) + float(y2)) / 2
                print(f"\n2D BBox center: ({cx_2d:.1f}, {cy_2d:.1f})")

# -----------------------------browse dataset function------------------------------
def browse_dataset(dataset_root="rebar_tying/datasets", scene_name="scene_incline"):
    import scipy.io as sio
    scene_path = os.path.join(dataset_root, "scenes", scene_name)
    assert os.path.isdir(scene_path), f"Scene not found: {scene_path}"

    # give the list of frames
    meta_dir = os.path.join(scene_path, "meta")
    frames = sorted([int(os.path.splitext(os.path.basename(p))[0]) 
                     for p in glob.glob(os.path.join(meta_dir, "*.mat"))])
    assert frames, "No frames found."

    state = {"fi": 0, "oi": 0, "posesN": 0, "quit": False}

    def load_geoms(frame_idx, obj_idx):
        meta = sio.loadmat(os.path.join(scene_path, "meta", f"{frame_idx:04d}.mat"))
        poses = np.asarray(meta["poses"]).astype(np.float32)
        K = np.asarray(meta["intrinsic_matrix"]).astype(np.float32).reshape(3,3)
        state["posesN"] = poses.shape[0]
        obj_idx = max(0, min(obj_idx, state["posesN"]-1))

        pose = poses[obj_idx]
        keypoint_3d = pose[:3, 3]
        normal_3d = pose[:3, 2]

        pc_file = os.path.join(scene_path, "pointclouds", f"{frame_idx:04d}_obj{obj_idx}.npy")
        assert os.path.exists(pc_file), f"Point cloud not found: {pc_file}"
        pointcloud = np.load(pc_file)

        geoms = []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.paint_uniform_color([0.7, 0.7, 0.7]); geoms.append(pcd)

        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sph.translate(keypoint_3d); sph.paint_uniform_color([1,0,0]); geoms.append(sph)

        normal_len = 0.1
        normal_end = keypoint_3d + normal_3d * normal_len
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([keypoint_3d, normal_end])
        line.lines  = o3d.utility.Vector2iVector([[0,1]])
        line.colors = o3d.utility.Vector3dVector([[0,0,1]])
        geoms.append(line)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(pose); geoms.append(frame)

        title = f"{scene_name}  frame={frame_idx:04d}  obj={obj_idx}/{state['posesN']-1}"
        return geoms, title

    # 建一个可更新的窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1200, height=900, window_name="Viewer")

    def refresh():
        vis.clear_geometries()
        geoms, title = load_geoms(frames[state["fi"]], state["oi"])
        for g in geoms: vis.add_geometry(g)
        vis.get_render_option().background_color = np.array([1,1,1])  # 白底可改
        # Set camera view to see Z-up coordinate system properly
        vis.get_view_control().set_front([1,0,0])  # Camera looks from X direction
        vis.get_view_control().set_up([0,0,1])     # Z-up in world frame
        vis.update_renderer()
        vis.poll_events()

    # Generate RGB overlay for current object
    def save_rgb_overlay(frame_idx, obj_idx):
        """Generate and save RGB overlay for current object"""
        try:
            meta = sio.loadmat(os.path.join(scene_path, "meta", f"{frame_idx:04d}.mat"))
            poses = np.asarray(meta["poses"]).astype(np.float32)
            K = np.asarray(meta["intrinsic_matrix"]).astype(np.float32).reshape(3,3)
            
            pose = poses[obj_idx]
            rgb_file = os.path.join(scene_path, 'rgb', f"{frame_idx:04d}.png")
            rgb_image = cv2.imread(rgb_file) if os.path.exists(rgb_file) else None
            if rgb_image is None:
                return
            
            labels_file = os.path.join(scene_path, 'labels_xyxy_depth', f"{frame_idx:04d}.txt")
            if not os.path.exists(labels_file):
                return
            
            # Load bbox
            with open(labels_file, 'r') as f:
                lines = f.readlines()
            if obj_idx < len(lines):
                parts = lines[obj_idx].strip().split()
                _, x1, y1, x2, y2, conf = parts
                x1, y1, x2, y2 = map(lambda x: int(round(float(x))), [x1, y1, x2, y2])
                
                # Draw bbox
                rgb_vis = rgb_image.copy()
                cv2.rectangle(rgb_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
                cx_2d = int((x1 + x2) * 0.5)
                cy_2d = int((y1 + y2) * 0.5)
                cv2.circle(rgb_vis, (cx_2d, cy_2d), 4, (0, 0, 255), -1)  # Red
                cv2.putText(rgb_vis, f"obj {obj_idx} conf={float(conf):.2f}", 
                           (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                cv2.putText(rgb_vis, f"obj {obj_idx} conf={float(conf):.2f}", 
                           (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # Regenerate point cloud using bbox mode (for better visualization)
                depth_file = os.path.join(scene_path, 'depth', f"{frame_idx:04d}.png")
                depth_image = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED) if os.path.exists(depth_file) else None
                if depth_image is not None:
                    bbox = [x1, y1, x2, y2]
                    pointcloud = depth_to_pointcloud_patch(depth_image, rgb_image, bbox, K, mode="bbox")
                    
                    # Project pointcloud
                    uv = project_3d_to_2d(pointcloud, K)
                    h, w = rgb_vis.shape[:2]
                    mask = (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
                    uv_int = uv[mask].astype(np.int32)
                    for (u, v) in uv_int:
                        rgb_vis[v, u] = (255, 0, 0)  # Blue points (BGR)
                
                # Save
                output_dir = os.path.join(dataset_root, "visualizations", scene_name)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"keypoint_f{frame_idx:04d}_o{obj_idx}.png")
                cv2.imwrite(output_file, rgb_vis)
                print(f"\n✓ Saved RGB overlay: {output_file}")
        except Exception as e:
            print(f"Error saving RGB overlay: {e}")
    
    # Track if we've already saved for current object
    saved_for_current = False
    
    # 键位：C 下一对象 -> 下一帧；B 上一对象/帧；O 仅切换对象；Q 退出
    def on_c(vis_):
        nonlocal saved_for_current
        # Save RGB overlay for current object before moving
        if not saved_for_current:
            save_rgb_overlay(frames[state["fi"]], state["oi"])
            saved_for_current = True
        
        if state["oi"] + 1 < state["posesN"]:
            state["oi"] += 1
        else:
            state["oi"] = 0
            state["fi"] = (state["fi"] + 1) % len(frames)
        saved_for_current = False  # Reset flag for next object
        refresh(); return False

    def on_o(vis_):
        state["oi"] = (state["oi"] + 1) % max(1, state["posesN"])
        refresh(); return False

    def on_b(vis_):
        if state["oi"] > 0:
            state["oi"] -= 1
        else:
            state["fi"] = (state["fi"] - 1) % len(frames)
            state["oi"] = 0
        refresh(); return False

    def on_q(vis_):
        state["quit"] = True
        return True  # 让窗口关闭

    vis.register_key_callback(ord('C'), on_c)
    vis.register_key_callback(ord('O'), on_o)
    vis.register_key_callback(ord('B'), on_b)
    vis.register_key_callback(ord('Q'), on_q)
    vis.register_key_callback(256, on_q)  # ESC

    refresh()
    while not state["quit"]:
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()

# 把 main() 改成调用浏览器：
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default="rebar_tying/datasets")
    ap.add_argument("--scene", default="scene_incline")
    args = ap.parse_args()
    browse_dataset(args.dataset_root, args.scene)
    
if __name__ == "__main__":
    main()