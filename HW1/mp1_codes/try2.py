import numpy as np
import matplotlib.pyplot as plt
from imageio import imread

# You could pip install the following dependencies if any is missing
# pip install -r requirements.txt

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0
keypoints_im = np.array([(604.593078169188, 583.1361439828671),
                       (1715.3135416380655, 776.304920238324),
                       (1087.5150188078305, 1051.9034760165837),
                       (79.20731171576836, 642.2524505093215)])

print(keypoints_im)
plt.clf()
plt.imshow(im)
plt.scatter(keypoints_im[:, 0], keypoints_im[:, 1])
plt.plot(keypoints_im[[0, 1, 2, 3, 0], 0], keypoints_im[[0, 1, 2, 3, 0], 1], 'g')

for ind, corner in enumerate(keypoints_im):
		plt.text(corner[0] + 30.0, corner[1] + 30.0, '#'+str(ind),
             c='b', family='sans-serif', size='x-large')
plt.title("Target Image and Keypoints")
plt.show()

'''
Question 1: specify the corners' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis
Output:
     - corners_court: a numpy array (4x2 matrix)
'''
# --------------------------- Begin your code here ---------------------------------------------

corners_court = np.array([[0,15.24],[28.65,15.24],[28.65,0],[0,0]])

# --------------------------- End your code here   ---------------------------------------------

'''
Question 2: complete the findHomography function
Arguments:
     pts_src - Each row corresponds to an actual point on the 2D plane (Nx2 matrix)
     pts_dst - Each row is the pixel location in the target image coordinate (Nx2 matrix)
Returns:
     H - The homography matrix (3x3 matrix)

Hints:
    - you might find the functions vstack, hstack to be handy for getting homogenous coordinates;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''
def findHomography(pts_src, pts_dst):
# --------------------------- Begin your code here ---------------------------------------------
    a = np.zeros(shape=(8, 9))
    for i in range(4):
        a[2 * i][0] = 0
        a[2 * i][1] = 0
        a[2 * i][2] = 0
        a[2 * i][3] = pts_src[i][0]
        a[2 * i][4] = pts_src[i][1]
        a[2 * i][5] = 1
        a[2 * i][6] = -1 * pts_dst[i][1] * pts_src[i][0]
        a[2 * i][7] = -1 * pts_dst[i][1] * pts_src[i][1]
        a[2 * i][8] = -1 * pts_dst[i][1]
        a[2 * i + 1][0] = pts_src[i][0]
        a[2 * i + 1][1] = pts_src[i][1]
        a[2 * i + 1][2] = 1
        a[2 * i + 1][3] = 0
        a[2 * i + 1][4] = 0
        a[2 * i + 1][5] = 0
        a[2 * i + 1][6] = -1 * pts_dst[i][0] * pts_src[i][0]
        a[2 * i + 1][7] = -1 * pts_dst[i][0] * pts_src[i][1]
        a[2 * i + 1][8] = -1 * pts_dst[i][0]
    U, S, V = np.linalg.svd(a)
    projection_matrix = V[len(V) - 1].reshape(3, 3)
    projection_matrix = projection_matrix / projection_matrix[2][2]
    return projection_matrix
# --------------------------- End your code here   ---------------------------------------------

# Calculate the homography matrix using your implementation
H = findHomography(corners_court, keypoints_im)


'''
Question 3.a: insert the logo virtually onto the state farm center image.
Specific requirements:
     - the size of the logo needs to be 3x6 meters;
     - the bottom left logo corner is at the location (23, 2.5) on the basketball court.
Returns:
     transform_target - The transformation matrix from logo.png image coordinate to target.png coordinate (3x3 matrix)

Hints:
     - Consider calculating the transform as the composition of the two: H_logo_target = H_court_target @ H_logo_court
     - Given the banner size in meters and image size in pixels, could you scale the logo image coordinate from pixels to meters?
     - What transform will move the logo to the target location?
     - Could you leverage the homography between basketball court to target image we computed in Q.2?
     - Image coordinate is y-down ((0, 0) at bottom-left corner) while we expect the inserted logo to be y-up, how would you handle this?
'''

# Read the banner image that we want to insert to the basketball court
logo = imread('logo.png') / 255.0
plt.clf()
plt.imshow(logo)
plt.title("Banner")
plt.show()

# --------------------------- Begin your code here ---------------------------------------------

logo_coner = np.array([[0,0],[1000,0],[1000,500],[0,500]])
coner_in_coord = np.array([[23,5.5],[29,5.5],[29,2.5],[23,2.5]])
H_lc = findHomography(logo_coner, coner_in_coord)
target_transform = np.dot(H,H_lc)

# --------------------------- End your code here   ---------------------------------------------

'''
Question 3.b: compute the warpImage function
Arguments:
     image - the source image you may want to warp (Hs x Ws x 4 matrix, R,G,B,alpha)
     H - the homography transform from the source to the target image coordinate (3x3 matrix)
     shape - a tuple of the target image shape (Wt, Ht)
Returns:
     image_warped - the warped image (Ht x Wt x 4 matrix)

Hints:
    - you might find the function numpy.meshgrid and numpy.ravel_multi_index useful;
    - are you able to get rid of any for-loop over all pixels?
    - directly calling warpAffine or warpPerspective in cv2 will receive zero point, but you could use as sanity-check of your own implementation
'''

def warpImage(img, M, shape):
  # --------------------------- Begin your code here ---------------------------------------------
# def warpImage(img, M, shape):
  # Create the meshgrid
  x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

  indices = np.vstack((x.flatten(), y.flatten(), np.ones(x.size, dtype=int)))

  # Apply the transformation
  transformed_indices = np.dot(M, indices)  # .astype(np.int32)
  x = transformed_indices[0, :]
  y = transformed_indices[1, :]
  scale = transformed_indices[2, :]
  x = (x / scale).astype(np.int32)
  y = (y / scale).astype(np.int32)

  # # Clip the x and y values to avoid out-of-bounds errors
  x = np.clip(x, 0, shape[1] - 1)
  y = np.clip(y, 0, shape[0] - 1)

  # Create the output image
  warped_img = np.zeros(shape=(shape[0], shape[1], 4), dtype=img.dtype)
  # warped_img = original_img
  warped_img[y, x] = img[indices[1], indices[0]]

  return warped_img
  # --------------------------- End your code here   ---------------------------------------------

# call the warpImage function
logo_warp = warpImage(logo, target_transform, im.shape)

plt.clf()
plt.imshow(logo_warp)
plt.title("Warped Banner")
plt.show()

'''
Question 3.c: alpha-blend the warped logo and state farm center image

im = logo * alpha_logo + target * (1 - alpha_logo)

Hints:
    - try to avoid for-loop. You could either use numpy's tensor broadcasting or explicitly call np.repeat / np.tile
'''

# --------------------------- Begin your code here ---------------------------------------------

def image_3c_warp(img, M, original_img, shape, alpha_logo):
    # Create the meshgrid
    x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    indices = np.vstack((x.flatten(), y.flatten(), np.ones(x.size,dtype = int)))

    # Apply the transformation
    transformed_indices = np.dot(M, indices) #.astype(np.int32)
    x = transformed_indices[0, :]
    y = transformed_indices[1, :]
    scale = transformed_indices[2, :]
    x = (x / scale).astype(np.int32)
    y = (y / scale).astype(np.int32)

    # # Clip the x and y values to avoid out-of-bounds errors
    x = np.clip(x, 0, shape[1] - 1)
    y = np.clip(y, 0, shape[0] - 1)

    # Create the output image、
    # warped_img = np.zeros(shape = (shape[0],shape[1],4), dtype=img.dtype)
    warped_img = original_img
    warped_img[y, x] = img[indices[1], indices[0]]* alpha_logo + warped_img[y, x] * (1 - alpha_logo)

    return warped_img

im = image_3c_warp(logo, target_transform, im, im.shape,alpha_logo=0.5)

# --------------------------- End your code here   ---------------------------------------------

plt.clf()
plt.imshow(im)
plt.title("Blended Image")
plt.show()

# convert im to uint8 to reduce file size
im *= 255
im = im.astype(np.uint8)

# dump the results for autograde
outfile = 'solution_homography.npz'
np.savez(outfile, corners_court, H, target_transform, logo_warp, im)