import copy

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import cv2

# read intrinsics, extrinsincs and camera images
K1 = np.load('assets/fountain/Ks/0005.npy')
K2 = np.load('assets/fountain/Ks/0004.npy')
R1 = np.load('assets/fountain/Rs/0005.npy')
R2 = np.load('assets/fountain/Rs/0004.npy')
t1 = np.load('assets/fountain/ts/0005.npy')
t2 = np.load('assets/fountain/ts/0004.npy')
img1 = cv2.imread('assets/fountain/images/0005.png')
img2 = cv2.imread('assets/fountain/images/0004.png')
h, w, _ = img1.shape

# resize the image to reduce computation
scale = 8 # you could try different scale parameters, e.g. 4 for better quality & slower speed.
img1 = cv2.resize(img1, (w//scale, h//scale))
img2 = cv2.resize(img2, (w//scale, h//scale))
h, w, _ = img1.shape

# visualize the left and right image
plt.figure()
# opencv default color order is BGR instead of RGB so we need to take care of it when visualization
plt.imshow(cv2.cvtColor(np.concatenate((img1, img2), axis=1), cv2.COLOR_BGR2RGB))
plt.title("Before rectification")
plt.show()

# Q6.a: How do the intrinsics change before and after the scaling?
# You only need to modify K1 and K2 here, if necessary. If you think they remain the same, leave here as blank and explain why.
# --------------------------- Begin your code here ---------------------------------------------
# print("K1:", K1)
K1 = K1 / scale
K1[2][2] = 1
K2 = K2 / scale
K2[2][2] = 1
# K1 = np.concatenate((K1[:2,:]/ scale, K1[2:, :]), axis = 0)
# K2 = np.concatenate((K2[:2,:]/ scale, K2[2:, :]), axis = 0)
# --------------------------- End your code here   ---------------------------------------------

# Compute the relative pose between two cameras
T1 = np.eye(4)
T1[:3, :3] = R1
T1[:3, 3:] = t1
T2 = np.eye(4)
T2[:3, :3] = R2
T2[:3, 3:] = t2
T = T2.dot(np.linalg.inv(T1)) # c1 to world and world to c2
R = T[:3, :3]
t = T[:3, 3:]

# Rectify stereo image pair such that they are frontal parallel. Here we call cv2 to help us
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1, None, K2, None,(w // 4, h // 4), R, t, 1, newImageSize=(0,0))
left_map  = cv2.initUndistortRectifyMap(K1, None, R1, P1, (w, h), cv2.CV_16SC2)
right_map = cv2.initUndistortRectifyMap(K2, None, R2, P2, (w, h), cv2.CV_16SC2)
left_img = cv2.remap(img1, left_map[0],left_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
right_img = cv2.remap(img2, right_map[0],right_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
plt.figure()
plt.imshow(cv2.cvtColor(np.concatenate((left_img, right_img), axis = 1), cv2.COLOR_BGR2RGB))
plt.title("After stereo rectification")
plt.show()

# Visualize images after rectification and report K1, K2 in your PDF report.


def stereo_matching_ssd(left_im, right_im, max_disp = 128, block_size = 7, method= 'SSD'):
  """
  Using sum of square difference to compute stereo matching.
  Arguments:
      left_im: left image (h x w x 3 numpy array)
      right_im: right image (h x w x 3 numpy array)
      max_disp: maximum possible disparity
      block_size: size of the block for computing matching cost
  Returns:
      disp_im: disparity image (h x w numpy array), storing the disparity values
  """
  # --------------------------- Begin your code here ---------------------------------------------
  disp_map = np.zeros_like(left_im[:, :, 0])
  depth = np.zeros_like(left_im[:, :, 0])
  _block_size = int((block_size-1)/2) # 3
  # Loop through each pixel in the left image
  for i in range(_block_size, left_im.shape[0] - _block_size): # 3 ~ m-3
    for j in range(_block_size, left_im.shape[1] - block_size): # 3 ~ n-3

      # Get the patch in the left image
      patch_left = left_im[i - _block_size:i + _block_size + 1, j - _block_size:j + _block_size + 1, :]
      # Initialize variables for storing the best disparity and minimum SSD
      best_disparity = 0
      min_ssd = float('inf')

      # Loop through a range of disparity values
      # method1:
      for d in range(max_disp):
        if j - _block_size - d < 0 or j + _block_size + 1 - d > right_im.shape[1] - 1:
          ssd = float('inf')
        else:
          patch_right = right_im[i - _block_size:i + _block_size + 1, j - _block_size - d:j + _block_size + 1 - d, :]
          #method1:
          if method == 'SSD':
              ssd = np.sum((patch_left - patch_right) ** 2)
          elif method == 'SAD':
              ssd = np.sum(abs(patch_left - patch_right))
          elif method == 'normalized_correlation':
              ssd = 1 - np.mean(np.multiply((patch_left - np.mean(patch_left)), (patch_right - np.mean(patch_right)))) / ((np.std(patch_left) * np.std(patch_right)) + 0.00001)
      # # # #method2:
      # for d in range(-max_disp, max_disp + 1):
      #   # Get the corresponding patch in the right image
      #   if j - _block_size + d < 0 or j + _block_size + 1 + d > right_im.shape[1] - 1:
      #     ssd = float('inf')
      #   else:
      #     patch_right = right_im[i - _block_size:i + _block_size + 1, j - _block_size + d:j + _block_size + 1 + d, :]
      #     ssd = np.sum((patch_left - patch_right) ** 2)


        # If the current disparity has a lower SSD, update the best disparity and min_ssd
        if ssd < min_ssd:
          best_disparity = abs(d)
          # print("best_disparity:", best_disparity)
          min_ssd = ssd

      # Store the best disparity for the current pixel
      disp_map[i, j] = best_disparity

      if best_disparity != 0:
        depth[i][j] = 5.0/abs(best_disparity)

  # --------------------------- End your code here   ---------------------------------------------
  return disp_map, depth


disparity, depth = stereo_matching_ssd(left_img, right_img, max_disp = 128, block_size=7, method='SSD') #128 7
# Depending on your implementation, runtime could be a few minutes.
# Feel free to try different hyper-parameters, e.g. using a higher-resolution image, or a bigger block size. Do you see any difference?
# You could also try to exclude bad matches in various ways if you are interested.

plt.figure()
plt.imshow(disparity)
plt.title("Disparity map")
plt.show()

plt.figure()
plt.imshow(depth)
plt.title("depth map")
plt.show()

# Compare your method and an off the shelf CV2's stereo matching results.
# Please list a few directions which you think could improve your own results
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
plt.imshow(np.concatenate((left_gray, right_gray), axis = 1), 'gray')
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=7)
disparity_cv2 = stereo.compute(left_gray, right_gray) / 16.0
plt.imshow(np.concatenate((disparity, disparity_cv2), axis = 1))
plt.show()

# Visualize disparity map and comparison against disparity_cv2 in your report.
#
#
# Q6 Bonus:

# --------------------------- Begin your code here ---------------------------------------------
xyz = None
color = None
# --------------------------- method1 ---------------------------------------------
C1 = -np.linalg.inv(R1) @ t1
C2 = -np.linalg.inv(R2) @ t2
baseline = np.linalg.norm(C2 - C1)
focal_length = K1[0][0]
depth_map = np.where(disparity != 0, (focal_length * baseline) / disparity, np.inf)

y, x = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]

pts_2d_hom = np.vstack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))

pts_3d_hom = np.linalg.inv(K1) @ pts_2d_hom * depth.ravel()
pts_3d = pts_3d_hom[:3, :].T

# Convert camera coordinate system to world coordinate system
R_inv = np.linalg.inv(R)
T_inv = -R_inv @ np.array([baseline,0,0])
pts_3d_world = (R_inv @ pts_3d.T).T + T_inv.reshape((1, 3))
color = img1.reshape(-1, 3) / 255.0

point_clouds = o3d.geometry.PointCloud()

point_clouds.points = o3d.utility.Vector3dVector(pts_3d_world)
point_clouds.colors = o3d.utility.Vector3dVector(color)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_clouds)
vis.run()
vis.destroy_window()
# ------------------------------------------------------------------------

# --------------------------- method2 ---------------------------------------------
C1 = -np.linalg.inv(R1) @ t1
C2 = -np.linalg.inv(R2) @ t2
baseline = np.linalg.norm(C2 - C1)
focal_length = K1[0][0]
# depth_map = np.where(disparity != 0, (1.0 * focal_length * baseline) / disparity, np.inf)
depth = (1.0 * baseline) / (disparity + 0.001)
y, x = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]

pts_2d_hom = np.vstack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))

pts_3d_hom = np.linalg.inv(K1) @ pts_2d_hom * depth.ravel()
pts_3d = pts_3d_hom[:3, :].T

# Convert camera coordinate system to world coordinate system
R_inv = np.linalg.inv(R)
T_inv = -R_inv @ np.array([baseline,0,0])
pts_3d_world = (R_inv @ pts_3d.T).T + T_inv.reshape((1, 3))
color = img1.reshape(-1, 3) / 255.0

point_clouds = o3d.geometry.PointCloud()

point_clouds.points = o3d.utility.Vector3dVector(pts_3d_world)
point_clouds.colors = o3d.utility.Vector3dVector(color)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(point_clouds)
vis.run()
vis.destroy_window()
# ------------------------------------------------------------------------

# print("xyz:", xyz.shape)
# --------------------------- End your code here   ---------------------------------------------

# Hints:
# What is the focal length? How large is the stereo baseline?
# Convert disparity to depth
# Unproject image color and depth map to 3D point cloud
# You can use Open3D to visualize the colored point cloud

# if xyz is not None:
#   pcd = o3d.geometry.PointCloud()
#   pcd.points = o3d.utility.Vector3dVector(xyz)
#   pcd.colors = o3d.utility.Vector3dVector(color)
#   o3d.visualization.draw_geometries([pcd])