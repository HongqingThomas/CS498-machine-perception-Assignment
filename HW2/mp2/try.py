'''
Question 5. Triangulation
In this question we move to 3D.
You are given keypoint matching between two images, together with the camera intrinsic and extrinsic matrix.
Your task is to perform triangulation to restore the 3D coordinates of the key points.
In your PDF, please visualize the 3d points and camera poses in 3D from three different viewing perspectives.
'''
import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
all_good_matches = np.load('assets/all_good_matches.npy')

K1 = np.load('assets/fountain/Ks/0000.npy')
K2 = np.load('assets/fountain/Ks/0005.npy')

R1 = np.load('assets/fountain/Rs/0000.npy')
R2 = np.load('assets/fountain/Rs/0005.npy')

t1 = np.load('assets/fountain/ts/0000.npy')
t2 = np.load('assets/fountain/ts/0005.npy')

def triangulate(K1, K2, R1, R2, t1, t2, all_good_matches):
    """
    Arguments:
        K1: intrinsic matrix for image 1, dim: (3, 3)
        K2: intrinsic matrix for image 2, dim: (3, 3)
        R1: rotation matrix for image 1, dim: (3, 3)
        R2: rotation matrix for image 1, dim: (3, 3)
        t1: translation for image 1, dim: (3,)
        t2: translation for image 1, dim: (3,)
        all_good_matches:  dim: (#matches, 4)
    Returns:
        points_3d, dim: (#matches, 3)
    """
    points_3d = np.zeros(shape=(all_good_matches.shape[0], 3))
    # --------------------------- Begin your code here ---------------------------------------------
    def find_3d_point(x1, y1, x2, y2, projection_matrix1, projection_matrix2):
        a = np.zeros(shape=(4, 4))
        a[0][0] = y1 * projection_matrix1[2][0] - projection_matrix1[1][0]
        a[0][1] = y1 * projection_matrix1[2][1] - projection_matrix1[1][1]
        a[0][2] = y1 * projection_matrix1[2][2] - projection_matrix1[1][2]
        a[0][3] = y1 * projection_matrix1[2][3] - projection_matrix1[1][3]
        a[1][0] = projection_matrix1[0][0] - x1 * projection_matrix1[2][0]
        a[1][1] = projection_matrix1[0][1] - x1 * projection_matrix1[2][1]
        a[1][2] = projection_matrix1[0][2] - x1 * projection_matrix1[2][2]
        a[1][3] = projection_matrix1[0][3] - x1 * projection_matrix1[2][3]
        a[2][0] = y2 * projection_matrix2[2][0] - projection_matrix2[1][0]
        a[2][1] = y2 * projection_matrix2[2][1] - projection_matrix2[1][1]
        a[2][2] = y2 * projection_matrix2[2][2] - projection_matrix2[1][2]
        a[2][3] = y2 * projection_matrix2[2][3] - projection_matrix2[1][3]
        a[3][0] = projection_matrix2[0][0] - x2 * projection_matrix2[2][0]
        a[3][1] = projection_matrix2[0][1] - x2 * projection_matrix2[2][1]
        a[3][2] = projection_matrix2[0][2] - x2 * projection_matrix2[2][2]
        a[3][3] = projection_matrix2[0][3] - x2 * projection_matrix2[2][3]
        U, S, V = np.linalg.svd(a)
        point3d_homo = V[len(V) - 1].reshape(4, 1)
        x = point3d_homo[0] / point3d_homo[3]
        y = point3d_homo[1] / point3d_homo[3]
        z = point3d_homo[2] / point3d_homo[3]
        return x, y, z

    Rt1 = np.concatenate((R1, t1), axis = 1)
    Projection_matrix1 = np.dot(K1, Rt1)
    Rt2 = np.concatenate((R2, t2), axis = 1)
    Projection_matrix2 = np.dot(K2, Rt2)

    for i in range(all_good_matches.shape[0]):
        [x1, y1, x2, y2] = all_good_matches[i]
        points_3d[i] = find_3d_point(x1, y1, x2, y2, Projection_matrix1, Projection_matrix2)
    # --------------------------- End your code here   ---------------------------------------------
    return points_3d

points_3d = triangulate(K1, K2, R1, R2, t1, t2, all_good_matches)
# print("points_3d: ", points_3d)
if points_3d is not None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)

    # Visualize both point and camera
    # Check this link for Open3D visualizer http://www.open3d.org/docs/release/tutorial/visualization/visualization.html#Function-draw_geometries
    # Check this function for adding a virtual camera in the visualizer http://www.open3d.org/docs/release/python_api/open3d.geometry.LineSet.html#open3d.geometry.LineSet.create_camera_visualization
    # Open3D is not the only option. You could use matplotlib, vtk or other visualization tools as well.
    # --------------------------- Begin your code here ---------------------------------------------

    # --------------------------- End your code here   ---------------------------------------------