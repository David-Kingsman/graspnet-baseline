'''
This script is used to fit a cylinder to a point cloud.
'''
import open3d as o3d
import numpy as np
import cv2
from sklearn.cluster import DBSCAN # cluster algorithm
from sklearn.preprocessing import StandardScaler # standardization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # principal component analysis
import time

# -----------------------------camera intrinsic parameters------------------------------
fx = 1734.7572357650336
fy = 1734.593101527403
cx = 632.2360387060742
cy = 504.996466076361

# -----------------------------depth and rgb image path------------------------------
depth_path = r'/home/zekaijin/graspnet-baseline/rebar_tying/texture_suppression_model/images/rebar_joint_pose_estimation/Nano0711/Vertical/v_p1_45_400_0_depth_image.tiff'
rgb_path = r'/home/zekaijin/graspnet-baseline/rebar_tying/texture_suppression_model/images/rebar_joint_pose_estimation/Nano0711/Vertical/v_p1_45_400_0_depth_filtered_image.jpg'

# depth_path = r'image\i_p1_69_400_0_depth_image.tiff'
# rgb_path = r'image\i_p1_69_400_0_depth_filtered_image.jpg'

# -----------------------------read image------------------------------
depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
rgb_raw = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
rgb_display = rgb_raw.copy()

# -----------------------------mouse click to select point------------------------------
clicked_point = []
def mouse_callback(event, x, y, flags, param):
    '''mouse callback function'''
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point.clear()
        clicked_point.append((x,y))
        print(f"Clicked point: ({x},{y})")
        cv2.destroyAllWindows()

cv2.imshow("Click on RGB Image", rgb_display)
cv2.setMouseCallback("Click on RGB Image", mouse_callback)
cv2.waitKey(0)
if not clicked_point:
    print("No point clicked, exit.")
    exit()

# -----------------------------generate local point cloud------------------------------
x, y = clicked_point[0]
h, w = depth_raw.shape
half_win = 100
x_min, x_max = max(x - half_win, 0), min(x + half_win, w)
y_min, y_max = max(y - half_win, 0), min(y + half_win, h)

depth_crop = depth_raw[y_min:y_max, x_min:x_max]
rgb_crop = rgb_raw[y_min:y_max, x_min:x_max]

xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
z = depth_crop
x3d = (xx - cx) * z / fx
y3d = (yy - cy) * z / fy

valid = (z > 0) & (z < 0.5)
xyz = np.stack((x3d[valid], y3d[valid], z[valid]), axis=-1)
colors = rgb_crop.reshape(-1, 3)[valid.flatten()] / 255.0

# -----------------------------build point cloud------------------------------
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd], window_name="point cloud display")

# -----------------------------denoising: remove statistical outliers------------------------------
pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
print(f"Original points: {len(pcd.points)}, After denoising: {len(pcd_clean.points)}")

if len(pcd_clean.points) == 0:
    print("No points remain after denoising! Exiting.")
    exit()

points = np.asarray(pcd_clean.points)

# -----------------------------point cloud downsampling function------------------------------
def downsample_points(points_np, voxel_size=0.005):
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(points_np)
    pcd_down = pcd_tmp.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd_down.points)

time1 = time.time()

# -----------------------------DBSCAN clustering------------------------------
print("DBSCAN clustering...")
points_scaled = StandardScaler().fit_transform(points)
db = DBSCAN(eps=0.3, min_samples=30).fit(points_scaled)
labels = db.labels_

unique_labels = set(labels)
print(f"Detected clusters (excluding noise): {len(unique_labels) - (1 if -1 in labels else 0)}")

colors_dbscan = plt.get_cmap("tab10")(labels % 10)[:, :3]
pcd_clustered = o3d.geometry.PointCloud()
pcd_clustered.points = o3d.utility.Vector3dVector(points)
pcd_clustered.colors = o3d.utility.Vector3dVector(colors_dbscan)
o3d.visualization.draw_geometries([pcd_clustered], window_name="Clustered PointCloud")

# -----------------------------extract the top three largest clusters------------------------------
import collections
label_counts = collections.Counter(labels[labels != -1])
top3_labels = [label for label, _ in label_counts.most_common(3)]

pts_class_1 = points[labels == top3_labels[0]]
pts_class_2_1 = points[labels == top3_labels[1]]
pts_class_2_2 = points[labels == top3_labels[2]]

# -----------------------------downsampling------------------------------
pts_class_1_ds = downsample_points(pts_class_1, voxel_size=0.001)
pts_class_2_1_ds = downsample_points(pts_class_2_1, voxel_size=0.001)
pts_class_2_2_ds = downsample_points(pts_class_2_2, voxel_size=0.001)

# -----------------------------merge the two rebar clusters------------------------------
pts_class_2_ds = np.vstack((pts_class_2_1_ds, pts_class_2_2_ds))

# -----------------------------cylinder fitting function (least squares)------------------------------
def fit_cylinder_least_squares(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    axis_dir = pca.components_[0]
    if axis_dir[2] < 0:
        axis_dir = -axis_dir

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

# -----------------------------create cylinder mesh------------------------------
def create_cylinder_mesh(axis_point, axis_dir, radius, h_min, h_max, extend_len=0, color=[1, 0, 0]):
    '''create cylinder mesh'''
    # extend height: extend extend_len at both ends
    height = (h_max - h_min) + 2 * extend_len

    mesh_cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh_cyl.compute_vertex_normals()
    mesh_cyl.paint_uniform_color(color)

    # calculate rotation matrix, align Z axis to axis_dir
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, axis_dir)
    s = np.linalg.norm(v)
    if s < 1e-6:
        R = np.eye(3)
    else:
        c = np.dot(z_axis, axis_dir)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    mesh_cyl.rotate(R, center=(0, 0, 0))

    # calculate axis center point (cylinder center)
    center_height = (h_min + h_max) / 2
    center_point = axis_point + axis_dir * (center_height - h_min)  # axis_point是在h_min位置

    # because the default center of the cylinder in Open3D is at the midpoint, and the height ranges from -height/2 to +height/2,
    # so just translate the cylinder to center_point
    mesh_cyl.translate(center_point)

    # axis line from center_point - height/2 * axis_dir to center_point + height/2 * axis_dir
    axis_line = o3d.geometry.LineSet()
    start_pt = center_point - axis_dir * (height / 2)
    end_pt = center_point + axis_dir * (height / 2)
    axis_line.points = o3d.utility.Vector3dVector([start_pt, end_pt])
    axis_line.lines = o3d.utility.Vector2iVector([[0, 1]])
    axis_line.colors = o3d.utility.Vector3dVector([color])

    return mesh_cyl, axis_line

# -----------------------------fit and generate mesh------------------------------
cyl1_point, cyl1_dir, cyl1_radius, cyl1_hmin, cyl1_hmax = fit_cylinder_least_squares(pts_class_1_ds)
print(cyl1_point, cyl1_dir, cyl1_radius)
cyl2_point, cyl2_dir, cyl2_radius, cyl2_hmin, cyl2_hmax = fit_cylinder_least_squares(pts_class_2_ds)
print(cyl2_point, cyl2_dir, cyl2_radius)

cyl1_mesh, cyl1_axis = create_cylinder_mesh(cyl1_point, cyl1_dir, cyl1_radius, cyl1_hmin, cyl1_hmax, color=[0, 0.75, 1.0])
cyl2_mesh, cyl2_axis = create_cylinder_mesh(cyl2_point, cyl2_dir, cyl2_radius, cyl2_hmin, cyl2_hmax, color=[0, 0.75, 1.0])

# -----------------------------visualize point cloud + fit cylinder------------------------------
pcd_1_vis = o3d.geometry.PointCloud()
pcd_1_vis.points = o3d.utility.Vector3dVector(pts_class_1_ds)
pcd_1_vis.paint_uniform_color([1, 0.5, 0])

pcd_2_1_vis = o3d.geometry.PointCloud()
pcd_2_1_vis.points = o3d.utility.Vector3dVector(pts_class_2_1_ds)
pcd_2_1_vis.paint_uniform_color([0, 0.5, 1])

pcd_2_2_vis = o3d.geometry.PointCloud()
pcd_2_2_vis.points = o3d.utility.Vector3dVector(pts_class_2_2_ds)
pcd_2_2_vis.paint_uniform_color([0, 0.5, 1])

pcd_1_vis = o3d.geometry.PointCloud()
pcd_1_vis.points = o3d.utility.Vector3dVector(pts_class_1_ds)
pcd_1_vis.paint_uniform_color([0.5, 0.5, 0.5])

pcd_2_1_vis = o3d.geometry.PointCloud()
pcd_2_1_vis.points = o3d.utility.Vector3dVector(pts_class_2_1_ds)
pcd_2_1_vis.paint_uniform_color([0.5, 0.5, 0.5])

pcd_2_2_vis = o3d.geometry.PointCloud()
pcd_2_2_vis.points = o3d.utility.Vector3dVector(pts_class_2_2_ds)
pcd_2_2_vis.paint_uniform_color([0.5, 0.5, 0.5])

# o3d.visualization.draw_geometries([pcd_1_vis, pcd_2_1_vis, pcd_2_2_vis, cyl1_mesh, cyl2_mesh, cyl1_axis, cyl2_axis],
#                                   window_name="Cylinder Fit Result")

# -----------------------------visualize point cloud + fit cylinder------------------------------
o3d.visualization.draw_geometries([pcd.paint_uniform_color([0.5, 0.5, 0.5]), cyl1_mesh, cyl2_mesh, cyl1_axis, cyl2_axis],
                                  window_name="Cylinder Fit Result")


# -----------------------------convert cylinder to point cloud and merge------------------------------
print("Converting cylinders to point clouds and merging...")
def mesh_to_pointcloud(mesh, num_points=3000):
    return mesh.sample_points_uniformly(number_of_points=num_points)

# 1. convert cylinder to point cloud and merge
cyl1_pcd = mesh_to_pointcloud(cyl1_mesh, num_points=3000)
cyl2_pcd = mesh_to_pointcloud(cyl2_mesh, num_points=3000)
cyl_combined_pcd = cyl1_pcd + cyl2_pcd

# 2. merge actual point cloud clusters
pcd_combined = pcd_1_vis + pcd_2_1_vis + pcd_2_2_vis

# 3. build KDTree for quick nearest neighbor search
pcd_combined_tree = o3d.geometry.KDTreeFlann(pcd_combined)
cyl_points = np.asarray(cyl_combined_pcd.points)

threshold = 0.003  # threshold distance, adjust based on actual data (unit: meter)

# 4. filter cylinder point cloud, only keep points within threshold distance from actual point cloud
indices_to_keep = []
for i, pt in enumerate(cyl_points):
    [k, idx, dist] = pcd_combined_tree.search_knn_vector_3d(pt, 1)
    if k > 0 and dist[0] < threshold**2:
        indices_to_keep.append(i)

filtered_cyl_points = cyl_points[indices_to_keep]

print(f"Original cylinder points: {len(cyl_points)}, after filtering: {len(filtered_cyl_points)}")

filtered_cyl_pcd = o3d.geometry.PointCloud()
filtered_cyl_pcd.points = o3d.utility.Vector3dVector(filtered_cyl_points)

# 5. similarly, filter actual point cloud, only keep points near the cylinder point cloud (optional)

# 6. downsample point cloudDBSCAN
voxel_size = 0.001
filtered_cyl_pcd_down = filtered_cyl_pcd.voxel_down_sample(voxel_size)
pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size)

# 7. ICP registration
reg = o3d.pipelines.registration.registration_icp(
    filtered_cyl_pcd_down, pcd_combined_down,
    max_correspondence_distance=voxel_size * 2,
    init=np.eye(4),
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

print("ICP fitness:", reg.fitness)
print("ICP inlier RMSE:", reg.inlier_rmse)
print("ICP transformation:\n", reg.transformation)

# 8. apply transformation for visualization
filtered_cyl_pcd_down.transform(reg.transformation)

o3d.visualization.draw_geometries([
    pcd_combined_down.paint_uniform_color([0.7, 0.7, 0.7]),
    filtered_cyl_pcd_down.paint_uniform_color([1, 0, 0])
], window_name="ICP Registration After Filtering")

time2 = time.time()

# -----------------------------visualize ICP registration after filtering------------------------------
o3d.visualization.draw_geometries([
    pcd.paint_uniform_color([0.5, 0.5, 0.5]),
    filtered_cyl_pcd_down.paint_uniform_color([0, 0.75, 1.0])
], window_name="ICP Registration After Filtering")

# print(time2 - time1, reg.fitness)