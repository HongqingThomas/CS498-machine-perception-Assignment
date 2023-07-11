# import lxr as love

import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
import open3d as o3d
import pyrender
import trimesh

# Load the image and plot the keypoints
im = imread('uiuc.png') / 255.0

# Read eight keypoints, including four court corners and four backboard corners
keypoints_im = np.array([
 [ 642.89378381,  589.79713627],
 [1715.31354164,  773.80704813],
 [1087.51501881, 1049.40560391],
 [  74.2115675 ,  637.2567063 ],
 [ 375.62146838,  464.07090689],
 [ 439.73351912,  462.40565882],
 [ 441.39876719,  496.54324428],
 [ 376.45409242,  499.87374042]
 ])

plt.figure()
plt.imshow(im)
plt.scatter(keypoints_im[:, 0], keypoints_im[:, 1])
plt.plot(keypoints_im[[0, 1, 2, 3, 0], 0], keypoints_im[[0, 1, 2, 3, 0], 1], 'g')
plt.plot(keypoints_im[[0+4, 1+4, 2+4, 3+4, 0+4], 0], keypoints_im[[0+4, 1+4, 2+4, 3+4, 0+4], 1], 'g')
for ind, corner in enumerate(keypoints_im):
		plt.text(corner[0] + 30.0, corner[1] + 30.0, '#'+str(ind),
						 c='b', family='sans-serif', size='x-large')
plt.title("Keypoints")
plt.show()



'''
Question 4: specify the keypoints' coordinates
Take point 3 as origin, the long edge as x axis and short edge as y axis,
upward direction perpendicular to the ground as z axis
Output:
		 - corners_3D: a numpy array (8x3 matrix)
'''

# Predefined constants on basketball court
lower_rim = 3.05 - 0.305 # height of backboard's lower rim
backboard_width = 1.83
backboard_height = 1.22
court_length = 28.65
court_width = 15.24
board_to_baseline = 1.22 # board to baseline distance

# --------------------------- Begin your code here ---------------------------------------------
corners_3d = [[0, court_width, 0],
              [court_length, court_width, 0],
              [court_length, 0, 0],
              [0,0,0],
              [board_to_baseline, (court_width - backboard_width)/2, lower_rim + backboard_height],
              [board_to_baseline, (court_width + backboard_width)/2, lower_rim + backboard_height],
              [board_to_baseline, (court_width + backboard_width)/2, lower_rim],
              [board_to_baseline, (court_width - backboard_width)/2, lower_rim],
              ]
# --------------------------- End your code here   ---------------------------------------------

'''
Question 5: complete the findProjection function
Arguments:
     xyz - Each row corresponds to an actual point in 3D with homogeneous coordinate (Nx4 matrix)
     uv - Each row is the pixel location in the homogeneous image coordinate (Nx3 matrix)
Returns:
     P - The projection matrix (4x3 matrix) such that uv = P @ xyz

Hints:
    - you might find the function vstack, hstack to be handy for getting homogenous coordinate;
    - you might find numpy.linalg.svd to be useful for solving linear system
    - directly calling findHomography in cv2 will receive zero point, but you could use it as sanity-check of your own implementation
'''
def findProjection(xyz, uv):
# --------------------------- Begin your code here ---------------------------------------------
    a = np.zeros(shape = (16,12))
    for i in range(8):
        a[2 * i][0] = 0
        a[2 * i][1] = 0
        a[2 * i][2] = 0
        a[2 * i][3] = 0
        a[2 * i][4] = xyz[i][0]
        a[2 * i][5] = xyz[i][1]
        a[2 * i][6] = xyz[i][2]
        a[2 * i][7] = 1
        a[2 * i][8] = -1 * uv[i][1] * xyz[i][0]
        a[2 * i][9] = -1 * uv[i][1] * xyz[i][1]
        a[2 * i][10] = -1 * uv[i][1] * xyz[i][2]
        a[2 * i][11] = -1 * uv[i][1]
        a[2 * i + 1][0] = xyz[i][0]
        a[2 * i + 1][1] = xyz[i][1]
        a[2 * i + 1][2] = xyz[i][2]
        a[2 * i + 1][3] = 1
        a[2 * i + 1][4] = 0
        a[2 * i + 1][5] = 0
        a[2 * i + 1][6] = 0
        a[2 * i + 1][7] = 0
        a[2 * i + 1][8] = -1 * uv[i][0] * xyz[i][0]
        a[2 * i + 1][9] = -1 * uv[i][0] * xyz[i][1]
        a[2 * i + 1][10] = -1 * uv[i][0] * xyz[i][2]
        a[2 * i + 1][11] = -1 * uv[i][0]
    # print("a:",a)
    U, S, V = np.linalg.svd(a)
    projection_matrix = V[len(V) - 1].reshape(3,4)
    projection_matrix = projection_matrix/projection_matrix[2][3]
    return projection_matrix

# --------------------------- End your code here   ---------------------------------------------

# Get homogeneous coordinate (using np concatenate)
uv = np.concatenate([keypoints_im, np.ones((len(keypoints_im), 1))], axis = 1)
xyz = np.concatenate([corners_3d, np.ones((len(corners_3d), 1))], axis = 1)


# Find the projection matrix from correspondences
P = findProjection(xyz, uv)
# Recalculate the projected point location
uv_project = P.dot(xyz.T).T
uv_project = uv_project / np.expand_dims(uv_project[:, 2], axis = 1)

# Plot reprojection.
plt.clf()
plt.imshow(im)
plt.scatter(uv[:, 0], uv[:, 1], c='r', label = 'original keypoints')
plt.scatter(uv_project[:, 0], uv_project[:, 1], c='b', label = 'reprojected keypoints')
plt.title('Reprojection')
plt.legend()
plt.show()



# Load the stanford bunny 3D mesh
bunny = o3d.io.read_triangle_mesh('./bunny.ply')
bunny.compute_vertex_normals()
# Today we will only consider using its vertices
verts = np.array(bunny.vertices)
verts_original = verts
'''
Question 6: project the stanford bunny onto the center of the basketball court

Output:
		- bunny_uv: all the vertices on image coordinate (35947x2 matrix)

Hints:
    - Transform the bunny from its object-centric 3D coordinate to basketball court 3D coordinate;
    - Make sure the bunny is above the ground
    - Do not forget to use homomgeneous coordinate for projection
'''

# --------------------------- Begin your code here ---------------------------------------------
Homo_verts = np.concatenate([verts, np.ones((len(verts), 1))], axis = 1)
Homo_verts[:,0] = Homo_verts[:,0] + court_length/2 - ((np.sum(Homo_verts[:,0]))/Homo_verts.shape[0])
Homo_verts[:,1] = Homo_verts[:,1] + court_width/2 - ((np.sum(Homo_verts[:,1]))/Homo_verts.shape[0])
Homo_verts[:,2] = Homo_verts[:,2] - np.min(Homo_verts[:,2])
bunny_uv = np.dot(P,Homo_verts.T)
bunny_uv[0,:] = bunny_uv[0,:] / bunny_uv[2,:]
bunny_uv[1,:] = bunny_uv[1,:] / bunny_uv[2,:]
bunny_uv = bunny_uv[0:2,:].T
# bunny_uv = []

# --------------------------- End your code here   ---------------------------------------------


# Visualize the Projection
plt.clf()
plt.imshow(im)
plt.scatter(bunny_uv[:, 0], bunny_uv[:, 1], c='b', s = 0.01, label = 'bunny')
plt.title('Stanford Bunny on State Farm Center')
plt.legend()
plt.show()

# Dump the results for autograde
outfile = 'solution_perspective.npz'
np.savez(outfile, corners_3d, P, bunny_uv)


'''
bonus points
Try to use an off-the-shelf renderer to render the 3D mesh in a more realistic
manner (e.g. pyrender, moderngl, blender, mitsuba2). 
'''

# --------------------------- Begin your code here ---------------------------------------------
verts = verts_original
verts[:,0] = verts[:,0] - ((np.sum(verts[:,0]))/verts.shape[0])
verts[:,1] = verts[:,1] - ((np.sum(verts[:,1]))/verts.shape[0])
triangles = np.array(bunny.triangles)

tm = trimesh.creation.icosahedron()
tm.vertices = verts
tm.faces = triangles
mesh = pyrender.Mesh.from_trimesh(tm)
# mesh = pyrender.Mesh.from_points(verts, colors=(200,200,200))

# compose scene
scene = pyrender.Scene(ambient_light=[1, 1, 3], bg_color=[20, 20, 20])
camera = pyrender.OrthographicCamera(xmag=2.5, ymag=2.5)
light = pyrender.DirectionalLight(color=[100,100,100], intensity=2e1)
scene.add(mesh, pose= np.eye(4))
scene.add(light, pose= np.eye(4))

def rotation_matrix(x_angle = 0, y_angle = 0, z_angle = 0):
    transform_matrix = np.dot(np.dot(np.array([[np.cos(z_angle), -1 * np.sin(z_angle), 0],[np.sin(z_angle), np.cos(z_angle),0],[0, 0, 1]]),
                                     np.array([[np.cos(y_angle), 0, np.sin(y_angle)],[0, 1, 0],[-1 * np.sin(y_angle), 0, np.cos(y_angle)]])),
                              np.array([[1, 0, 0],[0, np.cos(x_angle), -1 * np.sin(x_angle)],[0, np.sin(x_angle), np.cos(x_angle)]]))
    return transform_matrix

transform_matrix = rotation_matrix(x_angle = (np.pi/4), y_angle = 0, z_angle = np.pi/4)
scene.add(camera, pose=[[transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], 2.5],
                        [transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], -2.5],
                        [transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], 5],
                        [ 0, 0, 0, 1]])

# render scene
r = pyrender.OffscreenRenderer(viewport_width=640,viewport_height=480,point_size=2.0)
color, _ = r.render(scene)
plt.figure(figsize=(8,8))
plt.imshow(color)
plt.show()
# --------------------------- End your code here   ---------------------------------------------