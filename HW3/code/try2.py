print("Full registration ...")
voxel_size = 0.02
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 1.5
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
      if t_id == s_id + 1:  # odometry case
        odometry = pred_poses[target_id]
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(
                np.linalg.inv(odometry)))
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    final_Ts[-1],
                                                    uncertain=False))
      elif t_id <= s_id + 3:  # loop closure case
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                    target_id,
                                                    final_Ts[-1],
                                                    uncertain=False))
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine,
    edge_prune_threshold=0.25,
    reference_node=0)
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        # option)