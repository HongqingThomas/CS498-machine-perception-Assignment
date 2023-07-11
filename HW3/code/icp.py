import copy
from re import I
import time
import cv2
import numpy as np
import open3d as o3d
import time
from sklearn.neighbors import KDTree
import transforms3d as transform

# Question 4: deal with point_to_plane = True

def fit_rigid(src, tgt, normal_tgt, point_to_plane):
    # Question 2: Rigid Transform Fitting
    # Implement this function
    # -------------------------
    T = np.identity(4)
    # assume each row represents coordinate of a point
    if point_to_plane == False:
        centroid_src = np.mean(src, axis=0)
        centroid_tgt = np.mean(tgt, axis=0)
        centered_src = src - centroid_src
        centered_tgt = tgt - centroid_tgt
        # print("centroid_src:", centroid_src.shape)
        # covariance_matrix = np.dot(centered_src.T, centered_tgt)
        covariance_matrix = np.dot(centered_tgt.T, centered_src)
        U, _, V = np.linalg.svd(covariance_matrix)

        # R = np.dot(U,V.T)
        R = np.dot(U, V)
        # R = np.dot(V.T, U.T)
        t = centroid_tgt - np.dot(R, centroid_src)
        # RtR: [[1.00000000e+00 - 1.64452575e-16 - 3.07641091e-16]
        #       [-1.64452575e-16  1.00000000e+00 - 2.29237585e-16]
        #       [-3.07641091e-16 - 2.29237585e-16 1.00000000e+00]]
        # t: [1.18285614 0.99242399 1.66882667]
        T[:3,:3] = R
        T[:3,3] = t.T
    else:
        C = np.matrix(np.zeros((6, 6)))
        b = np.matrix(np.zeros((6, 1)))
        for i in range(src.shape[0]):
            p = np.matrix(src[i, :])
            q = np.matrix(tgt[i, :])
            n = np.matrix(normal_tgt[i, :])
            c = np.cross(p, n).T
            C[:3, :3] += c @ c.T
            C[:3, 3:] += c @ n
            C[3:, :3] += n.T @ c.T
            C[3:, 3:] += n.T @ n
            b[:3, 0] += c @ (p-q) @ n.T
            b[3:, 0] += n.T @ (p-q) @ n.T
        x = -np.dot(np.linalg.pinv(C), b)
        T[:3,:3] = transform.euler.euler2mat(x[0], x[1], x[2])
        T[0, 3] = x[3]
        T[1, 3] = x[4]
        T[2, 3] = x[5]
    # -------------------------
    return T


# Question 4: deal with point_to_plane = True
def icp(source, target, init_pose=np.eye(4), max_iter = 20, point_to_plane = False):
    src = np.asarray(source.points)#.T
    tgt = np.asarray(target.points)#.T
    tgt_normals = np.asarray(target.normals)

    # ---------------------------------------------------
    T = init_pose
    transforms = []
    delta_Ts = []
    inlier_ratio = 0
    print("iter %d: inlier ratio: %.2f" % (0, inlier_ratio))

    traget_tree = KDTree(tgt)

    for i in range(max_iter):

        distances, indices = traget_tree.query(src, k=1)
        indices = np.squeeze(indices.reshape(1,-1))

        T_delta = fit_rigid(src, tgt[indices], tgt_normals[indices], point_to_plane)
        src = np.dot(src, T_delta[:3, :3].T) + T_delta[:3, 3]
        T = T @ T_delta
        # T = T_delta @ T
        threshold = 0.05
        inliers = np.count_nonzero(distances < threshold)
        inlier_ratio = np.sum(inliers) / distances.shape[0]
        # ---------------------------------------------------

        print("iter %d: inlier ratio: %.2f" % (i + 1, inlier_ratio))
        # relative update from each iteration
        delta_Ts.append(T_delta.copy())
        # pose estimation after each iteration
        transforms.append(T.copy())

        if inlier_ratio > 0.999:
            break

    return transforms, delta_Ts

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
    normalized_y = (y - K[1, 2]) / K[0, 0]

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

def pose_error(estimated_pose, gt_pose):
    # Question 5: Translation and Rotation Error
    # Use equations 5-6 in https://cmp.felk.cvut.cz/~hodanto2/data/hodan2016evaluation.pdf
    # Your implementation between the lines
    # ---------------------------
    error = np.zeros(2)
    estimated_R = estimated_pose[:3, :3]
    estimated_t = estimated_pose[:3, 3]
    gt_R = gt_pose[:3, :3]
    gt_t = gt_pose[:3, 3]
    error[1] = np.sqrt(sum((gt_t - estimated_t) ** 2))
    error[0] = np.arccos((np.trace(np.dot(estimated_R, gt_R.T)) - 1) / 2)
    # ---------------------------
    return error

def read_data(ind = 0):
    K = np.loadtxt("data/camera-intrinsics.txt", delimiter=' ')
    depth_im = cv2.imread("data/frame-%06d.depth.png"%(ind),-1).astype(float)
    depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
    depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
    T = np.loadtxt("data/frame-%06d.pose.txt"%(ind))  # 4x4 rigid transformation matrix
    color_im = cv2.imread("data/frame-%06d.color.jpg"%(ind),-1)
    color_im = cv2.cvtColor(color_im, cv2.COLOR_BGR2RGB)  / 255.0
    return color_im, depth_im, K, T

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
    final_Ts, delta_Ts = icp(source, target,point_to_plane = False)

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