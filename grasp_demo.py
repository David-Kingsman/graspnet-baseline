import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# ==================== hard code parameters ====================
CHECKPOINT_PATH = '/home/zekaijin/graspnet-baseline/logs/log_rs/checkpoint.tar'  # 修改为你的权重路径
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01
DATA_DIR = '/home/zekaijin/graspnet-baseline/doc/example_test/'  # 修改为你的数据目录

# camera intrinsic (use depth camera parameters to generate point cloud)
DEPTH_INTR = {
    "ppx": 319.304,  # cx
    "ppy": 236.915,  # cy
    "fx": 387.897,  # fx
    "fy": 387.897  # fy
}
DEPTH_FACTOR = 1000.0  # depth factor, adjust according to actual data 1 (this is based on the camera used in the experiment)


# ==================== network definition ====================
def get_net():
    net = GraspNet(
        input_feature_dim=0,
        num_view=NUM_VIEW,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


# ==================== data processing ====================
def get_and_process_data(data_dir):
    # automatically find color and depth files
    color_files = [f for f in os.listdir(data_dir) if f.endswith('_color.png') or f == 'color.png']
    depth_files = [f for f in os.listdir(data_dir) if f.endswith('_depth_graspnet.png') or f == 'depth.png']
    
    if not color_files or not depth_files:
        raise FileNotFoundError(f"color or depth file not found in {data_dir}")
    
    color_file = os.path.join(data_dir, color_files[0])
    depth_file = os.path.join(data_dir, depth_files[0])
    
    print(f"using color file: {color_file}")
    print(f"using depth file: {depth_file}")
    
    color = np.array(Image.open(color_file), dtype=np.float32) / 255.0

    # load depth image and print information
    depth_img = Image.open(depth_file)
    depth = np.array(depth_img)

    # print("\n=== depth image analysis ===")
    # print("image format:", depth_img.format)
    # print("storage mode:", depth_img.mode)
    # print("NumPy array shape:", depth.shape)
    # print("data type:", depth.dtype)
    # print("minimum value:", np.min(depth))
    # print("maximum value:", np.max(depth))
    # print("non-zero pixel count:", np.count_nonzero(depth))
    # print("zero value pixel ratio: %.2f%%" % (100 * (1 - np.count_nonzero(depth) / depth.size)))

    # # depth factor analysis suggestion
    # max_depth = np.max(depth)
    # suggested_factors = []
    # if max_depth > 10:  # if the maximum value is large, it may be stored in millimeters
    #     suggested_factors.append(1000)  # millimeters to meters
    # if max_depth < 10:  # if the value is small, it may be stored in floating point units
    #     suggested_factors.append(1.0)

    # print("\ndepth factor suggestion:")
    # print(f"current used depth factor: {DEPTH_FACTOR}")
    # if suggested_factors:
    #     print("detected possible depth factors:")
    #     for f in suggested_factors:
    #         print(f"-> {f} (explanation: {'millimeters to meters' if f == 1000 else 'directly use meters'}")
    # else:
    #     print("cannot automatically infer depth factor, please verify manually")

    # other processing remains the same...
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'mask1.png')))

    # verify image size
    print("\n=== size verification ===")
    print("depth image size:", depth.shape[::-1])  # (width, height)
    print("color image size:", color.shape[:2][::-1])
    print("camera parameter preset size:", (1280, 720))

    # create camera parameter object 
    camera = CameraInfo(
        width=1280,
        height=720,
        fx=DEPTH_INTR['fx'],
        fy=DEPTH_INTR['fy'],
        cx=DEPTH_INTR['ppx'],
        cy=DEPTH_INTR['ppy'],
        scale=DEPTH_FACTOR
    )

    # generate point cloud using depth image only
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # apply mask to the point cloud
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # point cloud sampling to get NUM_POINT points
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]

    # convert to Open3D point cloud (for visualization) and convert to tensor
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    # convert to tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)

    end_points = {'point_clouds': cloud_sampled}
    return end_points, cloud_o3d


# ==================== collision detection ====================
def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]


# ==================== print grasp poses ====================
def print_grasp_poses(gg):
    print(f"\nTotal grasps after collision detection: {len(gg)}")
    for i, grasp in enumerate(gg):
        print(f"\nGrasp {i + 1}:")
        print(f"Position (x,y,z): {grasp.translation}")
        print(f"Rotation Matrix:\n{grasp.rotation_matrix}")
        print(f"Score: {grasp.score:.4f}")
        print(f"Width: {grasp.width:.4f}")


# ==================== main process ====================
def demo(data_dir):
    # initialize network
    net = get_net()

    # process data
    end_points, cloud_o3d = get_and_process_data(data_dir)

    # forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy())

    # collision detection
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, np.asarray(cloud_o3d.points))

    # print grasp poses
    print_grasp_poses(gg)

    # visualization
    gg.nms().sort_by_score()
    gg = gg[:50]  # take the top 50 grasps for visualization
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud_o3d, *grippers])


if __name__ == '__main__':
    demo(DATA_DIR)