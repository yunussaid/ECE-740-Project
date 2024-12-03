import cv2
import numpy as np
import glob

# Checkerboard dimensions (inner corners)
checkerboard_rows = 5
checkerboard_cols = 8
square_size = 30  # Size of one square in consistent units (e.g., mm)

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
object_points = np.zeros((checkerboard_rows * checkerboard_cols, 3), np.float32)
object_points[:, :2] = np.mgrid[0:checkerboard_cols, 0:checkerboard_rows].T.reshape(-1, 2) * square_size

# Arrays to store object points and image points
obj_points = []
img_points_left = []
img_points_right = []

# Load and sort stereo images
left_images = sorted(glob.glob('Final image group/left_*.jpg'))
right_images = sorted(glob.glob('Final image group/right_*.jpg'))

if len(left_images) != len(right_images) or len(left_images) == 0:
    print("Error: Mismatched or no images found.")
    exit()

for left_img_path, right_img_path in zip(left_images, right_images):
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(gray_left, (checkerboard_cols, checkerboard_rows), None)
    ret_right, corners_right = cv2.findChessboardCorners(gray_right, (checkerboard_cols, checkerboard_rows), None)

    if ret_left and ret_right:
        obj_points.append(object_points)

        # Refine corner locations
        corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)

        img_points_left.append(corners_left)
        img_points_right.append(corners_right)

        # Display chessboard detection for debugging
        cv2.drawChessboardCorners(img_left, (checkerboard_cols, checkerboard_rows), corners_left, ret_left)
        cv2.drawChessboardCorners(img_right, (checkerboard_cols, checkerboard_rows), corners_right, ret_right)
        cv2.imshow('Left Camera', img_left)
        cv2.imshow('Right Camera', img_right)
        cv2.waitKey(500)
    else:
        print(f"Chessboard not detected in pair: {left_img_path}, {right_img_path}")

cv2.destroyAllWindows()

# Perform stereo calibration
if len(obj_points) > 0:
    ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        img_points_left,
        img_points_right,
        None, None, None, None,
        gray_left.shape[::-1],
        flags=(cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_RATIONAL_MODEL),
        criteria=criteria
    )

    print("Camera 1 Matrix (K1):\n", K1, "\n")
    print("Camera 2 Matrix (K2):\n", K2, "\n")
    print("Distortion Coefficients for Camera 1 (D1):\n", D1, "\n")
    print("Distortion Coefficients for Camera 2 (D2):\n", D2, "\n")
    print("Rotation Matrix (R):\n", R, "\n")
    print("Translation Vector (T):\n", T, "\n")
else:
    print("Error: No valid image pairs for calibration.")


# Save stereo calibration parameters to a .npz file
baseline = np.linalg.norm(T)  # Baseline is the norm of the translation vector T

np.savez(
    'stereo_params.npz',
    K1=K1,  # Intrinsic matrix of the left camera
    D1=D1,  # Distortion coefficients of the left camera
    K2=K2,  # Intrinsic matrix of the right camera
    D2=D2,  # Distortion coefficients of the right camera
    R=R,  # Rotation matrix between the two cameras
    T=T,  # Translation vector between the two cameras
    E=E,  # Essential matrix
    F=F,  # Fundamental matrix
    baseline=baseline,  # Distance between the cameras
    focal_length=K1[0, 0],  # Focal length in pixels (assumes fx is consistent for both cameras)
    principal_point=(K1[0, 2], K1[1, 2])  # Principal point (cx, cy) for the left camera
)

print("Stereo calibration parameters have been saved to 'stereo_params.npz'.")

