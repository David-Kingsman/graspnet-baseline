#!/usr/bin/env python3
"""
检查GT pose的一致性：验证坐标系定义是否统一
"""
import os
import numpy as np
import scipy.io as sio
import glob
from pathlib import Path
import open3d as o3d

def visualize_pose_axes(pose, length=0.1):
    """可视化pose的坐标系轴"""
    origin = pose[:3, 3]
    x_axis = pose[:3, 0] * length
    y_axis = pose[:3, 1] * length
    z_axis = pose[:3, 2] * length
    
    axes = o3d.geometry.LineSet()
    points = np.array([origin, origin + x_axis, origin + y_axis, origin + z_axis])
    lines = np.array([[0, 1], [0, 2], [0, 3]])  # x(red), y(green), z(blue)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes

def analyze_pose_statistics(dataset_root):
    """分析pose统计信息，检查坐标系一致性"""
    scenes_dir = os.path.join(dataset_root, 'scenes')
    scene_folders = sorted(glob.glob(os.path.join(scenes_dir, 'scene_*')))
    
    all_z_axes = []
    all_x_axes = []
    all_z_dirs = []  # z轴与[0,0,1]的点积（判断朝向）
    
    print("="*80)
    print("检查GT Pose坐标系一致性")
    print("="*80)
    
    for scene_path in scene_folders:  # 检查所有场景
        scene_name = os.path.basename(scene_path)
        meta_files = sorted(glob.glob(os.path.join(scene_path, 'meta', '*.mat')))
        
        print(f"\n【场景】{scene_name}: {len(meta_files)} frames")
        
        # 【改进】放宽抽样范围以稳定评估（检查前50帧或全量）
        meta_files_to_check = meta_files[:50] if len(meta_files) > 50 else meta_files
        for meta_file in meta_files_to_check:
            try:
                meta = sio.loadmat(meta_file)
                poses = meta['poses']  # (N, 4, 4)
                
                for pose in poses:
                    z_axis = pose[:3, 2]
                    x_axis = pose[:3, 0]
                    
                    all_z_axes.append(z_axis)
                    all_x_axes.append(x_axis)
                    # 检查z轴是否朝上（与[0,0,1]的点积）
                    z_up = np.dot(z_axis, np.array([0, 0, 1]))
                    all_z_dirs.append(z_up)
                    
            except Exception as e:
                print(f"  Error loading {meta_file}: {e}")
                continue
    
    if len(all_z_axes) == 0:
        print("❌ 没有找到pose数据！")
        return
    
    all_z_axes = np.array(all_z_axes)
    all_x_axes = np.array(all_x_axes)
    all_z_dirs = np.array(all_z_dirs)
    
    print(f"\n【统计结果】共检查 {len(all_z_axes)} 个pose")
    print(f"Z轴朝向统计：")
    print(f"  - 朝上 (dot([0,0,1]) > 0.5): {np.sum(all_z_dirs > 0.5)} ({100*np.sum(all_z_dirs > 0.5)/len(all_z_dirs):.1f}%)")
    print(f"  - 朝下 (dot([0,0,1]) < -0.5): {np.sum(all_z_dirs < -0.5)} ({100*np.sum(all_z_dirs < -0.5)/len(all_z_dirs):.1f}%)")
    print(f"  - 水平 (|dot| < 0.5): {np.sum(np.abs(all_z_dirs) < 0.5)} ({100*np.sum(np.abs(all_z_dirs) < 0.5)/len(all_z_dirs):.1f}%)")
    
    print(f"\nZ轴方向聚类分析：")
    z_means = np.mean(all_z_axes, axis=0)
    z_std = np.std(all_z_axes, axis=0)
    print(f"  - 均值: [{z_means[0]:.3f}, {z_means[1]:.3f}, {z_means[2]:.3f}]")
    print(f"  - 标准差: [{z_std[0]:.3f}, {z_std[1]:.3f}, {z_std[2]:.3f}]")
    
    # 如果标准差很大，说明方向不一致
    if np.max(z_std) > 0.5:
        print(f"\n⚠️  警告：Z轴方向标准差很大（{np.max(z_std):.3f}），说明坐标系定义不一致！")
    else:
        print(f"\n✅ Z轴方向相对一致（标准差 {np.max(z_std):.3f}）")
    
    # 检查是否有随机旋转（角度分布）
    if len(all_z_axes) > 10:
        # 计算第一个z轴与后续z轴的夹角
        ref_z = all_z_axes[0]
        angles = []
        for z in all_z_axes[1:]:
            dot = np.clip(np.dot(ref_z, z), -1, 1)
            angle_deg = np.arccos(dot) * 180 / np.pi
            angles.append(angle_deg)
        
        angles = np.array(angles)
        print(f"\n与第一个pose的Z轴夹角统计：")
        print(f"  - 均值: {np.mean(angles):.1f}°")
        print(f"  - 中位数: {np.median(angles):.1f}°")
        print(f"  - 90°以上: {np.sum(angles > 90)} ({100*np.sum(angles > 90)/len(angles):.1f}%)")
        
        if np.mean(angles) > 60:
            print(f"\n❌ 警告：平均夹角 {np.mean(angles):.1f}° 很大，说明坐标系定义混乱！")
            print(f"   这会导致模型无法学习（旋转误差≈120°）")

def visualize_sample_poses(dataset_root, num_samples=5):
    """可视化几个pose样本"""
    scenes_dir = os.path.join(dataset_root, 'scenes')
    scene_folders = sorted(glob.glob(os.path.join(scenes_dir, 'scene_*')))
    
    geometries = []
    
    for scene_path in scene_folders[:1]:  # 只看第一个场景
        meta_files = sorted(glob.glob(os.path.join(scene_path, 'meta', '*.mat')))
        
        for meta_file in meta_files[:num_samples]:
            try:
                meta = sio.loadmat(meta_file)
                poses = meta['poses']
                
                # 加载对应的点云
                frame_idx = int(Path(meta_file).stem)
                pc_file = glob.glob(os.path.join(scene_path, 'pointclouds', f"{frame_idx:04d}_obj*.npy"))[0]
                pc = np.load(pc_file)
                
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)
                pcd.paint_uniform_color([0.5, 0.5, 0.5])
                geometries.append(pcd)
                
                # 添加pose坐标轴
                for pose in poses:
                    axes = visualize_pose_axes(pose, length=0.05)
                    geometries.append(axes)
                    
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    if len(geometries) > 0:
        print(f"\n可视化 {len(geometries)} 个对象...")
        o3d.visualization.draw_geometries(geometries, window_name="Pose一致性检查")
    else:
        print("没有找到可可视化的数据")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default="rebar_tying/datasets")
    parser.add_argument("--viz", action="store_true", help="可视化pose样本")
    args = parser.parse_args()
    
    analyze_pose_statistics(args.dataset_root)
    
    if args.viz:
        visualize_sample_poses(args.dataset_root)

