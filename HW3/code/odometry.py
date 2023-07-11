from re import I
import numpy as np
import open3d as o3d
from icp import read_data, icp, rgbd2pts


color_im, depth_im, K, T_init = read_data(0)
pcd_init = rgbd2pts(color_im, depth_im, K)
pcd_down = pcd_init.voxel_down_sample(voxel_size=0.02)
h, w, c = color_im.shape

step = 10 # try different step
end = 100
cam_init = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.1)
gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.1)
gt_cam.paint_uniform_color((1, 0, 0))

pred_poses = {}
gt_poses = {}
pcds = {}
pred_poses[0] = np.eye(4) # assuming first frame is global coordinate center
gt_poses[0] = np.eye(4) # assuming first frame is global coordinate center
pcds[0] = pcd_down # save point cloud for init frame
vis_list = [pcd_init, cam_init, gt_cam]

for frame in range(step, end, step):
    color_im, depth_im, K, T_tgt = read_data(frame - step)
    target = rgbd2pts(color_im, depth_im, K)
    color_im, depth_im, K, T_src = read_data(frame)
    source = rgbd2pts(color_im, depth_im, K)

    # some pre-processing, including computing normals and downsampling
    source_down = source.voxel_down_sample(voxel_size=0.02)
    target_down = target.voxel_down_sample(voxel_size=0.02)
    source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Question 6 --- Could you update the camera poses using ICP?
    # Hint1: you could call your ICP to estimate relative pose betwen frame t and t-step
    # Hint2: based on the estimated transform between frame t and t-step, are you able to compute world to camera transform for t?

    # Your code
    # ------------------------
    T_W2C = np.eye(4)  # world to current camera frame
    final_Ts, delta_Ts = icp(source_down, target_down, max_iter = 20, point_to_plane=False)
    estimated_transform = np.linalg.inv(final_Ts[-1])
    T_W2C = np.dot(pred_poses[frame - step], estimated_transform)
    pred_poses[frame] = T_W2C
    # ------------------------

    # get ground-truth pose
    T_W2C_gt =  np.linalg.inv(T_src) @ T_init # world to init to current    T_init：第一帧在大地坐标系里的位置
    gt_poses[frame] = T_W2C_gt # ground truth
    pcds[frame] = source_down

    current_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, T_W2C, scale = 0.1)
    gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, T_W2C_gt, scale = 0.1)
    gt_cam.paint_uniform_color((1, 0, 0))
    source.transform(np.linalg.inv(T_W2C))
    vis_list.append(source)
    vis_list.append(current_cam)
    vis_list.append(gt_cam)
    print("Frame %d is done" % frame)

o3d.visualization.draw_geometries(vis_list,
                                  zoom=0.422,
                                  front = [ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ],
                                  lookat = [ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ],
                                  up = [ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])


# Question 7: Relative Trajectory Error
# implement the relative rotation error and relative translation error
# reference, Eq.[2] and Eq.[3] in:
# https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR2012/data/papers/424_O3C-04.pdf
# please write down your derivations

# Your code
# ------------------------
error = np.zeros(2)
iter = 0
# for frame_i in range(step, end, step):
#     estimated_pi = pred_poses[frame_i]
#     gt_pi = gt_poses[frame_i]
#     for frame_j in range(step, end, step):
#         if frame_j != frame_i:
#             iter += 1
#             estimated_pj = pred_poses[frame_i]
#             gt_pj = gt_poses[frame_i]
#             composition_operation = np.linalg.inv(np.linalg.inv(estimated_pj) @ estimated_pi) @ (np.linalg.inv(gt_pj) @ gt_pi)
#             # error[0] += np.arccos((np.trace(np.dot((np.linalg.inv(estimated_pi) @ estimated_pj), (np.linalg.inv(gt_pi) @ gt_pj).T)) - 2) / 2)
#             error[0] += np.arccos((np.trace(composition_operation) - 2) / 2)
#             error[1] += np.sqrt(sum(composition_operation[:3,3] ** 2))
# error /= iter
for frame in range(step, end, step):
    estimated_pj = pred_poses[frame]
    gt_pj = gt_poses[frame]
    estimated_pi = pred_poses[frame - step]
    gt_pi = gt_poses[frame - step]
    composition_operation = np.linalg.inv(np.linalg.inv(estimated_pj) @ estimated_pi) @ (np.linalg.inv(gt_pj) @ gt_pi)
    # error[0] += np.arccos((np.trace(np.dot((np.linalg.inv(estimated_pi) @ estimated_pj), (np.linalg.inv(gt_pi) @ gt_pj).T)) - 2) / 2)
    error[0] += np.arccos((np.trace(composition_operation) - 2) / 2)
    error[1] += np.sqrt(sum(composition_operation[:3,3] ** 2))
    iter += 1
error /= iter
print("Rotation/Translation Error:", error)
# ------------------------


# Question 8: Pose graph optimization
# Now we have an simple odometry solution, where each frame's pose in world coordinate is decided by its previous frame's pose and a relative transformation provided by ICP.
# Given the pose T_0 and T_40, please validate whether the relative transformation
# calculated from the two odometry poses will perfectly agree with the transformation we estimated from ICP.
# If not, explain why. If yes, explain why (in your pdf).

color_im, depth_im, K, T_tgt = read_data(0)
target = rgbd2pts(color_im, depth_im, K)
color_im, depth_im, K, T_src = read_data(40)
source = rgbd2pts(color_im, depth_im, K)

# some pre-processing, including computing normals and downsampling
source_down = source.voxel_down_sample(voxel_size=0.02)
target_down = target.voxel_down_sample(voxel_size=0.02)
source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
final_Ts, delta_Ts = icp(source_down, target_down, max_iter=50, point_to_plane=False)

T_0 = pred_poses[0]
T_40 = pred_poses[40]
print("Relative transfrom from ICP:", np.linalg.inv(final_Ts[-1]))
print("Relative transfrom from odometry:", T_40)

# Question 8: to ensure the consistency, we could build a pose graph to further improve the performance.
# Each node is a camera pose
# Each edge will describe the relative transformation between the node, provided by ICP.
# The general idea of pose graph optimization is to jointly optimize the pose such that maximum consensus has been reached:
# argmin_{T_i} \sum_i^N \sum_{j>i}^{i+K} (T_ij - inv(T_j) @ T_i)^2
# where T_ij is transformation from i to j, T_i is transformation from i to world (global coordinate)

# In this question, you are going to leverage pose graph optimization to build dense pose graph.
# A node will be connected if their difference in frame number is smaller or equal to 30
# Open3D provides us with some helpful functions
# 0. Building pose graph pose_graph = o3d.pipelines.registration.PoseGraph()
# 1. Add one graph node: o3d.pipelines.registration.PoseGraphNode(init)
# 2. Add one graph edge: o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, tranform_from_icp)
# 3. Optimize pose graph: o3d.pipelines.registration.global_optimization()

# Hints:
# Be careful about the transformation, before we are outputing extrinsic matrix which is world to camera.
# Now each node records a transformation that goes from camera to world.

# Your code
# ------------------------

# pose_graph = o3d.pipelines.registration.PoseGraph()
#
# for i in range(len(pred_poses)):
#     node_pose = pred_poses[i * 10]
#     node = o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(node_pose))#  node_pose; np.linalg.inv(node_pose)
#     pose_graph.nodes.append(node)
#
# frame_list = list(pred_poses.keys())
# for i in range(len(pred_poses)-1):
#     for j in range(i+1, len(pred_poses)):
#         if j < i+4:
#             source_id = frame_list[i]
#             target_id = frame_list[j]
#             final_Ts, delta_Ts = icp(pcds[source_id], pcds[target_id], point_to_plane=False)
#             transform_from_icp = final_Ts[-1]
#             edge = o3d.pipelines.registration.PoseGraphEdge(i, j, transform_from_icp)
#             pose_graph.edges.append(edge)
#
# method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
# criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
# option = o3d.pipelines.registration.GlobalOptimizationOption()
# o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)


# pose_graph = o3d.pipelines.registration.PoseGraph()
#
# odometry = pred_poses[0]
# pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
#
# frame_list = list(pred_poses.keys())
# for i in range(len(pred_poses)):
#     for j in range(i+1, len(pred_poses)):
#         if j < i+4:
#             source_id = frame_list[i]
#             target_id = frame_list[j]
#             final_Ts, delta_Ts = icp(pcds[source_id], pcds[target_id], point_to_plane=False)
#             transform_from_icp = final_Ts[-1]
#             if j == i + 1:
#                 # odometry = np.dot(transform_from_icp, odometry)
#                 odometry = pred_poses[target_id]
#                 pose_graph.nodes.append(
#                     o3d.pipelines.registration.PoseGraphNode(
#                         np.linalg.inv(odometry)))
#                 pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j,
#                                                      transform_from_icp,
#                                                      uncertain=False))
#             else:
#                 pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(i, j,
#                                                                                  transform_from_icp,
#                                                                                  uncertain=True))
# max_correspondence_distance_fine = 0.03
# option = o3d.pipelines.registration.GlobalOptimizationOption(
#     max_correspondence_distance=max_correspondence_distance_fine,
#     edge_prune_threshold=0.25,
#     reference_node=0)
# o3d.pipelines.registration.global_optimization(
#     pose_graph,
#     o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
#     o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
#     option)


# good one
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
  pose_graph = o3d.pipelines.registration.PoseGraph()
  frame_list = list(pred_poses.keys())
  odometry = pred_poses[0]
  pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
  n_frames = len(frame_list)
  for s_id in range(n_frames):
    source_id = frame_list[s_id]
    for t_id in range(s_id + 1, n_frames):
      target_id = frame_list[t_id]
      final_Ts, delta_Ts = icp(pcds[source_id], pcds[target_id], point_to_plane=False)
      print("Build o3d.pipelines.registration.PoseGraph")
      if t_id == s_id + 1:
        odometry = pred_poses[target_id]
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    final_Ts[-1],
                                                    uncertain=False))
      elif t_id <= s_id + 3:
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    final_Ts[-1],
                                                    uncertain=True))

method = o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
criteria = o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria()
option = o3d.pipelines.registration.GlobalOptimizationOption()
o3d.pipelines.registration.global_optimization(pose_graph, method, criteria, option)




# print("pred_poses:", len(pred_poses))
frame_list = list(pred_poses.keys())

# ------------------------

print("Transform points and display")
vis_list = []
for point_id in range(len(pcds)):
    point_frame = frame_list[point_id]
    pcds[point_frame].transform(pose_graph.nodes[point_id].pose)
    T_C2W = pose_graph.nodes[point_id].pose
    pgo_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(T_C2W), scale = 0.1)
    pgo_cam.paint_uniform_color((0, 1, 0))
    gt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, gt_poses[point_frame], scale = 0.1)
    gt_cam.paint_uniform_color((1, 0, 0))
    odometry_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, pred_poses[point_frame], scale = 0.1)
    odometry_cam.paint_uniform_color((0, 0, 1))
    vis_list.append(pgo_cam)
    vis_list.append(odometry_cam)
    vis_list.append(gt_cam)
    vis_list.append(pcds[point_frame])

o3d.visualization.draw_geometries(vis_list,
                                  zoom=0.422,
                                  front = [ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ],
                                  lookat = [ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ],
                                  up = [ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])
