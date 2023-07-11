import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np
import open3d as o3d
import time
from sklearn.neighbors import KDTree

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i


def rgbd2pts(color_im, depth_im, K):
    # Question 1: unproject rgbd to color point cloud, provide visualization in your document
    # Your implementation between the lines
    # ---------------------------
    # N = 0 # todo
    # color = np.zeros((N, 3))
    # xyz = np.zeros((N, 3))

    height, width = depth_im.shape[0], depth_im.shape[1]
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the normalized image coordinates
    normalized_x = (x - K[0, 2]) / K[0, 0]
    normalized_y = (y - K[1, 2]) / K[1, 1]

    # Unproject the depth image to 3D points
    points_3d = np.dstack((normalized_x * depth_im, normalized_y * depth_im, depth_im))

    # Reshape the 3D points and the RGB image to create the colored point cloud
    xyz = points_3d.reshape(-1, 3)
    color = color_im.reshape(-1, 3)

    # ---------------------------

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd

def read_data(ind = 0):
  K = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
  depth_im = cv2.imread("data/frame-%06d.depth.png"%(ind),-1).astype(float)
  depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
  depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
  T = np.loadtxt("data/frame-%06d.pose.txt"%(ind))  # 4x4 rigid transformation matrix
  color_im = cv2.imread("data/frame-%06d.color.jpg"%(ind),-1)
  color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)  / 255.0
  return color_im, depth_im, K, T

def pose_error(estimated_pose, gt_pose):
  # Question 5: Translation and Rotation Error
  # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
  # Your implementation between the lines
  # ---------------------------
  error = 0
  # ---------------------------
  return error

if __name__ == "__main__":

  # pairwise ICP

  # read color, image data and the ground-truth, converting to point cloud
  color_im, depth_im, K, T_tgt = read_data(0)
  target = rgbd2pts(color_im, depth_im, K)
  color_im, depth_im, K, T_src = read_data(40)
  source = rgbd2pts(color_im, depth_im, K)

  # downsampling and normal estimatoin
  source = source.voxel_down_sample(voxel_size=0.02)
  target = target.voxel_down_sample(voxel_size=0.02)
  source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
  target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


  # # print("source:", np.asarray(source.points))
  # T = fit_rigid(np.asarray(source.points)[:10000,:], np.asarray(target.points)[:10000,:], point_to_plane=False)
  # print("T:", T)
  # # print("source:", np.asarray(source.points).shape) #source: (18732, 3)
  # # print("target:", np.asarray(target.points).shape) #target: (20784, 3)
  # # ValueError: shapes (3,18732) and (20784,3) not aligned: 18732 (dim 1) != 20784 (dim 0)
  # # source number and target number are not same


  # conduct ICP (your code)
  final_Ts, delta_Ts = icp(source, target)

  # visualization
  vis = o3d.visualization.Visualizer()
  vis.create_window()
  ctr = vis.get_view_control()
  ctr.set_front([ -0.11651295252277051, -0.047982289143896774, -0.99202945108647766 ])
  ctr.set_lookat([ 0.023592929264511786, 0.051808635289583765, 1.7903649529102956 ])
  ctr.set_up([ 0.097655832648056065, -0.9860023571949631, -0.13513952033284915 ])
  ctr.set_zoom(0.42199999999999971)
  vis.add_geometry(source)
  vis.add_geometry(target)

  save_image = False

  # update source images
  for i in range(len(delta_Ts)):
      source.transform(delta_Ts[i])
      vis.update_geometry(source)
      vis.poll_events()
      vis.update_renderer()
      time.sleep(0.2)
      if save_image:
          vis.capture_screen_image("temp_%04d.jpg" % i)

  # visualize camera
  h, w, c = color_im.shape
  tgt_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.eye(4), scale = 0.2)
  src_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(T_src) @ T_tgt, scale = 0.2)
  pred_cam = o3d.geometry.LineSet.create_camera_visualization(w, h, K, np.linalg.inv(final_Ts[-1]), scale = 0.2)

  gt_pose = np.linalg.inv(T_src) @ T_tgt
  pred_pose = np.linalg.inv(final_Ts[-1])
  p_error = pose_error(pred_pose, gt_pose)
  print("Ground truth pose:", gt_pose)
  print("Estimated pose:", pred_pose)
  print("Rotation/Translation Error", p_error)

  tgt_cam.paint_uniform_color((1, 0, 0))
  src_cam.paint_uniform_color((0, 1, 0))
  pred_cam.paint_uniform_color((0, 0.5, 0.5))
  vis.add_geometry(src_cam)
  vis.add_geometry(tgt_cam)
  vis.add_geometry(pred_cam)

  vis.run()
  vis.destroy_window()

  # Provide visualization of alignment with camera poses in write-up.
  # Print pred pose vs gt pose in write-up.