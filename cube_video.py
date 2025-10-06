import numpy as np
import cv2
from Project_1_CalCam import cam_cal

# Load your projection matrix M from previous calibration
M, K, R, t = cam_cal()  # If cam_cal returns (M, K, R, t)


# Define cube vertices (1x1x1 cube at origin)
cube_vertices = np.array([
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 1, 1],
    [0, 1, 1, 1]
], dtype=float)

# Define edges as pairs of vertex indices
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0), # bottom square
    (4, 5), (5, 6), (6, 7), (7, 4), # top square
    (0, 4), (1, 5), (2, 6), (3, 7)  # vertical edges
]

# Load background image
img_bg = cv2.imread('test_image.bmp')

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('cube_animation.avi', fourcc, 20, (img_bg.shape[1], img_bg.shape[0]))

# ...existing code...

num_frames = 60
half_frames = num_frames // 2

# Diagonal movement in X and Y
path_xy = np.linspace(0, 10, half_frames)
# Upward movement in Z
path_z = np.linspace(0, 10, num_frames - half_frames)

for frame in range(num_frames):
    img = img_bg.copy()
    moved_vertices = cube_vertices.copy()
    if frame < half_frames:
        # Move diagonally in X and Y
        t = path_xy[frame]
        moved_vertices[:, 0] += t  # X
        moved_vertices[:, 1] += t  # Y
    else:
        # Move up in Z
        t = path_z[frame - half_frames]
        moved_vertices[:, 2] += t  # Z

    # Project vertices
    projected = []
    for v in moved_vertices:
        v_proj = M @ v
        u = int(v_proj[0] / v_proj[2])
        v_ = int(v_proj[1] / v_proj[2])
        projected.append((u, v_))

    # Draw edges
    for i, j in cube_edges:
        pt1 = projected[i]
        pt2 = projected[j]
        cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    video.write(img)

video.release()
print("Video saved as cube_animation.avi")