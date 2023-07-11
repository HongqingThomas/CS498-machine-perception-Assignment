'''
Questions 2-4. Fundamental matrix estimation

Question 2. Eight-point Estimation
For this question, your task is to implement normalized and unnormalized eight-point algorithms to find out the fundamental matrix between two cameras.
We've provided a method to compute the average geometric distance, which is the distance between each projected keypoint from one image to its corresponding epipolar line in the other image.
You might consider reading that code below as a reminder for how we can use the fundamental matrix.
For more information on the normalized eight-point algorithm, please see this link: https://en.wikipedia.org/wiki/Eight-point_algorithm#Normalized_algorithm
Note that the normalized version may not necessarily perform better.

Question 3. RANSAC
Your task is to implement RANSAC to find out the fundamental matrix between two cameras if the correspondences are noisy.

Please report the average geometric distance based on your estimated fundamental matrix, given 1, 100, and 10000 iterations of RANSAC.
Please also visualize the inliers with your best estimated fundamental matrix in your solution for both images (we provide a visualization function).
In your PDF, please also explain why we do not perform SVD or do a least-square over all the matched key points.

Question 4. Visualizing Epipolar Lines
Please visualize the epipolar lines for both images for your estimated F in Q2 and Q3.

To draw on images, cv2.line, cv2.circle are useful to plot lines and circles.
Check our Lecture 4, Epipolar Geometry, to learn more about equation of epipolar line.
Our Lecture 4 and 5 cover most of the concepts here.
This link also gives a thorough review of epipolar geometry:
    https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
'''

import os
import cv2 # our tested version is 4.5.5
import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
import random
from pathlib import Path
import copy

basedir= Path('assets/fountain')
img1 = cv2.imread(str(basedir / 'images/0000.png'), 0)
img2 = cv2.imread(str(basedir /'images/0005.png'), 0)

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(img1, cmap='gray')
axarr[1].imshow(img2, cmap='gray')
plt.show()

# --------------------- Question 2

def calculate_geometric_distance(all_matches, F):
    """
    Calculate average geometric distance from each projected keypoint from one image to its corresponding epipolar line in another image.
    Note that you should take the average of the geometric distance in two direction (image 1 to 2, and image 2 to 1)
    Arguments:
        all_matches: all matched keypoint pairs that loaded from disk (#all_matches, 4).
        F: estimated fundamental matrix, (3, 3)
    Returns:
        average geometric distance.
    """
    ones = np.ones((all_matches.shape[0], 1))
    all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
    all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
    # Epipolar lines.
    F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
    F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
    # Geometric distances.
    p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
    p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
    d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
    d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

    # Final distance.
    dist1 = d1.sum() / all_matches.shape[0]
    dist2 = d2.sum() / all_matches.shape[0]

    dist = (dist1 + dist2)/2
    return dist

# Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4). Same pair of images as before
# For each row, it consists (k1_x, k1_y, k2_x, k2_y).
# If necessary, you can convert float to int to get the integer coordinate
eight_good_matches = np.load('assets/eight_good_matches.npy')
# print("eight_good_matches:", eight_good_matches[:,:])
# print(np.mean(eight_good_matches[:,:2], axis = 0))
all_good_matches = np.load('assets/all_good_matches.npy')

def estimate_fundamental_matrix(original_matches, normalize=False):
    """
    Arguments:
        matches: Coords of matched keypoint pairs in image 1 and 2, dims (#matches, 4).
        normalize: Boolean flag for using normalized or unnormalized alg.
    Returns:
        F: Fundamental matrix, dims (3, 3).
    """
    F = np.eye(3)
    # --------------------------- Begin your code here ---------------------------------------------
    def f_normalize(points):
        """Normalize the image points to improve numerical stability"""
        mean = np.mean(points, axis=0)
        dist = np.sqrt(np.sum((points - mean) ** 2, axis=1))
        scale = np.sqrt(2) / np.mean(dist)
        T = np.array([
            [scale, 0, -scale * mean[0]],
            [0, scale, -scale * mean[1]],
            [0, 0, 1]])
        normalized_points = np.dot(T, np.hstack((points, np.ones((points.shape[0], 1)))).T).T
        # print("normalized_points:", normalized_points)
        return normalized_points[:,:2], T

    matches = copy.deepcopy(original_matches)
    if normalize:
        # Normalized camera coordinate: everything in camera coordinate but z is normalized to 1
        matches[:,:2], T1 = f_normalize(matches[:,:2])
        matches[:,2:], T2 = f_normalize(matches[:,2:])
        # matches, T2 = f_normalize(matches)

    p1_u = matches[:,0]
    p1_v = matches[:,1]
    p2_u = matches[:,2]
    p2_v = matches[:,3]
    ones = np.ones(shape=(matches.shape[0],1))
    A = np.hstack((np.multiply(p1_u, p2_u).reshape((-1,1)),
                   np.multiply(p2_u, p1_v).reshape((-1,1)),
                   p2_u.reshape((-1,1)),
                   np.multiply(p2_v, p1_u).reshape((-1,1)),
                   np.multiply(p1_v, p2_v).reshape((-1,1)),
                   p2_v.reshape((-1,1)),
                   p1_u.reshape((-1,1)),
                   p1_v.reshape((-1,1)),
                   ones.reshape((-1,1))))

    _,_,V = np.linalg.svd(A)
    F_rank_3 = V[len(V)-1].reshape(3,3)
    U,S,V = np.linalg.svd(F_rank_3)
    S[2] = 0
    F_rank_2 = U @ np.diag(S) @ V

    if normalize:
        F = T2.T @ F_rank_2 @ T1
        # a = 1
    else:
        F = F_rank_2
    # --------------------------- End your code here   ---------------------------------------------
    return F

# print(eight_good_matches.shape)
F_with_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=True)
F_without_normalization = estimate_fundamental_matrix(eight_good_matches, normalize=False)

# Evaluation (these numbers should be quite small)
print(f"F_with_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_with_normalization)}")
print(f"F_without_normalization average geo distance: {calculate_geometric_distance(all_good_matches, F_without_normalization)}")

# --------------------- Question 3

def ransac(all_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold):
    """
    Arguments:
        all_matches: coords of matched keypoint pairs in image 1 and 2, dims (# matches, 4).
        num_iteration: total number of RANSAC iteration
        estimate_fundamental_matrix: your eight-point algorithm function but use normalized version
        inlier_threshold: threshold to decide if a point is inlier
    Returns:
        best_F: best Fundamental matrix, dims (3, 3).
        inlier_matches_with_best_F: (#inliers, 4)
        avg_geo_dis_with_best_F: float
    """

    best_F = np.eye(3)
    inlier_matches_with_best_F = None
    avg_geo_dis_with_best_F = 0.0
    m,n = all_matches.shape[0], all_matches.shape[1] # 200,4

    ite = 0
    prev_inliers_number = 0
    # --------------------------- Begin your code here ---------------------------------------------

    #while ite < num_iteration:
    while ite < num_iteration:
        ite += 1
        # random sample correspondences
        sample_index = random.sample(range(0,m),8) # eg: [131, 48, 119, 40, 100, 170, 134, 110]
        # print(sample_index)
        random_eight_good_matches = all_matches[sample_index]
        # estimate the minimal fundamental estimation problem
        F = estimate_fundamental_matrix(random_eight_good_matches, normalize=True)

        def calculate_distance(all_matches, F):
            ones = np.ones((all_matches.shape[0], 1))
            all_p1 = np.concatenate((all_matches[:, 0:2], ones), axis=1)
            all_p2 = np.concatenate((all_matches[:, 2:4], ones), axis=1)
            # Epipolar lines.
            F_p1 = np.dot(F, all_p1.T).T  # F*p1, dims [#points, 3].
            F_p2 = np.dot(F.T, all_p2.T).T  # (F^T)*p2, dims [#points, 3].
            # Geometric distances.
            p1_line2 = np.sum(all_p1 * F_p2, axis=1)[:, np.newaxis]
            p2_line1 = np.sum(all_p2 * F_p1, axis=1)[:, np.newaxis]
            d1 = np.absolute(p1_line2) / np.linalg.norm(F_p2, axis=1)[:, np.newaxis]
            d2 = np.absolute(p2_line1) / np.linalg.norm(F_p1, axis=1)[:, np.newaxis]

            # # Final distance.
            # dist1 = d1.sum() / all_matches.shape[0]
            # dist2 = d2.sum() / all_matches.shape[0]

            dist = (d1 + d2) / 2
            return dist

        dist = calculate_distance(all_matches, F).reshape(200) # shape:(200, )
        # compute # of inliers
        inliers_index = np.where(dist < inlier_threshold)
        # update the current best solution
        inliers_number = len(inliers_index[0])
        if inliers_number > prev_inliers_number:
            best_F = F
            prev_inliers_number = inliers_number
            inlier_matches_with_best_F = all_matches[inliers_index]
            avg_geo_dis_with_best_F = dist.sum() / all_matches.shape[0]
    # --------------------------- End your code here   ---------------------------------------------
    return best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F

def visualize_inliers(im1, im2, inlier_coords):
    for i, im in enumerate([im1, im2]):
        plt.subplot(1, 2, i+1)
        plt.imshow(im, cmap='gray')
        plt.scatter(inlier_coords[:, 2*i], inlier_coords[:, 2*i+1], marker="x", color="red", s=10)
    plt.show()

# num_iteration = 1
# inlier_threshold = 0.01 # TODO: change the inlier threshold by yourself
# best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold)


num_iterations = [1, 100, 10000]
inlier_threshold = 0.01 # TODO: change the inlier threshold by yourself
for num_iteration in num_iterations:
    best_F, inlier_matches_with_best_F, avg_geo_dis_with_best_F = ransac(all_good_matches, num_iteration, estimate_fundamental_matrix, inlier_threshold)
    if inlier_matches_with_best_F is not None:
        print(f"num_iterations: {num_iteration}; avg_geo_dis_with_best_F: {avg_geo_dis_with_best_F};")
        # print(inlier_matches_with_best_F)
        visualize_inliers(img1, img2, inlier_matches_with_best_F)

# --------------------- Question 4

def visualize(estimated_F, img1, img2, kp1, kp2):
    # --------------------------- Begin your code here ---------------------------------------------
    ones = np.ones(shape= (kp1.shape[0], 1))
    homo_kp1 = np.concatenate((kp1, ones), axis = 1)
    homo_kp2 = np.concatenate((kp2, ones), axis = 1)

    lines1 = np.dot(estimated_F, homo_kp1.T).T
    lines1 = lines1 / np.sqrt(lines1[:,0]**2 + lines1[:,1]**2)[:, np.newaxis]

    lines2 = np.dot(estimated_F.T, homo_kp2.T).T
    lines2 = lines2 / np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)[:, np.newaxis]

    # Draw the epilines on the images
    for line in lines1:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img1.shape[1], -(line[2] + line[0] * img1.shape[1]) / line[1]])
        cv2.line(img1, (x0, y0), (x1, y1), (0, 255, 0), 1)

    for line in lines2:
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])
        cv2.line(img2, (x0, y0), (x1, y1), (0, 255, 0), 1)

    # Show the images
    cv2.imshow('Image 1 with epipolar lines', img1)
    cv2.imshow('Image 2 with epipolar lines', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --------------------------- End your code here   ---------------------------------------------
    pass

all_good_matches = np.load('assets/all_good_matches.npy')
F_Q2 = F_with_normalization # link to your estimated F in Q2
F_Q3 = best_F # link to your estimated F in Q3
visualize(F_Q2, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])
visualize(F_Q3, img1, img2, all_good_matches[:, :2], all_good_matches[:, 2:])