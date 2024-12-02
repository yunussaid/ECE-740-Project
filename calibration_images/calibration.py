import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
checkerboard_rows = 5  # Adjust based on your pattern
checkerboard_cols = 8  # Adjust based on your pattern
square_size = 30  # Size of one square in millimeters (or other consistent unit)

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
object_points = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
object_points[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points
obj_points = []  # 3D points in real world space
img_points_left = []  # 2D points in left image plane
img_points_right = []  # 2D points in right image plane

# Load and sort stereo images
left_images = sorted(glob.glob('left_*.jpg'))  # Match all left images
right_images = sorted(glob.glob('right_*.jpg'))  # Match all right images

# Debugging: Print file lists
print("Left images:", left_images)
print("Right images:", right_images)

if len(left_images) != len(right_images) or len(left_images) == 0:
    print("Error: Mismatched or no images found.")
    exit()

# Process each image pair
for left_img_path, right_img_path in zip(left_images, right_images):
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows), None)

    # Debug: Visualize the detection
    if ret_left:
        cv2.drawChessboardCorners(img_left, (checkerboard_cols, checkerboard_rows), corners_left, ret_left)
        cv2.imshow('Left Camera - Chessboard', img_left)
    else:
        print(f"Chessboard not detected in LEFT image: {left_img_path}")

    if ret_right:
        cv2.drawChessboardCorners(img_right, (checkerboard_cols, checkerboard_rows), corners_right, ret_right)
        cv2.imshow('Right Camera - Chessboard', img_right)
    else:
        print(f"Chessboard not detected in RIGHT image: {right_img_path}")

    cv2.waitKey(500)  # Pause to display the results


cv2.destroyAllWindows()

# Check if enough valid pairs were found
if len(obj_points) == 0:
    print("Error: No valid chessboard pairs detected.")
    exit()

# Perform stereo calibration
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_left, img_points_right, gray_left.shape[::-1], None, None, None, None, flags=cv2.CALIB_FIX_INTRINSIC
)

# Output calibration results
print("Camera 1 Matrix (K1):", K1)
print("Camera 2 Matrix (K2):", K2)
print("Rotation Matrix (R):", R)
print("Translation Vector (T):", T)

# Save calibration parameters
np.savez('stereo_params.npz', K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
