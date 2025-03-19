# This file is for testing the calibration params (calibration_params.npz) acquired from calibration_params.py. It
# contains multiple tests that are somewhat redundant but they are left unoptimized to save on time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test 1 - Rectify calibration_images 02-pair and draw epipolar line for chessboard corners on pre-rectified image
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_1():

    # Load stereo parameters
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_params_path = os.path.join(curr_dir, 'calibration_params.npz')
    calibration_params = np.load(calibration_params_path)
    K1, D1, K2, D2 = calibration_params['K1'], calibration_params['D1'], calibration_params['K2'], calibration_params['D2']
    R, T, F = calibration_params['R'], calibration_params['T'], calibration_params['F']

    # Load stereo images (adjust paths as needed)
    left_image_path = os.path.join(curr_dir, 'calibration_images/left_02.jpg')
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_path = os.path.join(curr_dir, 'calibration_images/right_02.jpg')
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Rectification
    image_size = left_image.shape[::-1]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    left_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # Display rectified images with epipolar lines
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(left_rectified, cmap='gray')
    plt.title('Rectified Left Image')
    plt.subplot(1, 2, 2)
    plt.imshow(right_rectified, cmap='gray')
    plt.title('Rectified Right Image')
    plt.show()

    # Chessboard dimensions
    checkerboard_rows = 5
    checkerboard_cols = 8
    checkerboard_size = (checkerboard_cols, checkerboard_rows)

    # Epipolar Line Consistency Function
    def draw_epipolar_lines(image, lines, points):
        """Draw epipolar lines and corresponding points."""
        h, w = image.shape
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
        for r, pt in zip(lines, points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
            cv2.line(output, (x0, y0), (x1, y1), color, 1)
            cv2.circle(output, (int(pt[0][0]), int(pt[0][1])), 5, color, -1)  # Explicitly cast to int
        return output

    # Detect chessboard corners
    ret_left, corners_left = cv2.findChessboardCorners(left_image, checkerboard_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(right_image, checkerboard_size, None)

    if ret_left and ret_right:
        # Refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(left_image, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(right_image, corners_right, (11, 11), (-1, -1), criteria)

        # Compute epilines for each set of points
        epilines_left = cv2.computeCorrespondEpilines(corners_right.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
        epilines_right = cv2.computeCorrespondEpilines(corners_left.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

        # Draw epipolar lines
        left_with_epilines = draw_epipolar_lines(left_image, epilines_left, corners_left)
        right_with_epilines = draw_epipolar_lines(right_image, epilines_right, corners_right)

        # Display epipolar lines
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(left_with_epilines, cv2.COLOR_BGR2RGB))
        plt.title('Left Image with Epilines')
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(right_with_epilines, cv2.COLOR_BGR2RGB))
        plt.title('Right Image with Epilines')
        plt.show()

    else:
        print("Chessboard corners not detected in one or both images.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test 2 - Draw 5 (adjustable) epipolar lines on calibration_images 02-pair using goodFeaturesToTrack()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_2():

    # Load stereo parameters
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_params_path = os.path.join(curr_dir, 'calibration_params.npz')
    calibration_params = np.load(calibration_params_path)
    K1, D1, K2, D2 = calibration_params['K1'], calibration_params['D1'], calibration_params['K2'], calibration_params['D2']
    R, T, F = calibration_params['R'], calibration_params['T'], calibration_params['F']

    # Load stereo images (adjust paths as needed)
    left_image_path = os.path.join(curr_dir, 'calibration_images/left_02.jpg')
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_path = os.path.join(curr_dir, 'calibration_images/right_02.jpg')
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Rectification
    image_size = left_image.shape[::-1]
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    left_rectified = cv2.remap(left_image, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_image, map2x, map2y, cv2.INTER_LINEAR)

    # Epipolar Line Consistency Function
    def draw_epipolar_lines(image, lines, points):
        """Draw epipolar lines and corresponding points."""
        h, w = image.shape
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
        for r, pt in zip(lines, points):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [w, -(r[2] + r[0] * w) / r[1]])
            cv2.line(output, (x0, y0), (x1, y1), color, 1)
            cv2.circle(output, (int(pt[0][0]), int(pt[0][1])), 5, color, -1)  # Explicitly cast to int
        return output

    # Compute epilines
    features_left = cv2.goodFeaturesToTrack(left_image, 5, 0.01, 10)
    features_right = cv2.goodFeaturesToTrack(right_image, 5, 0.01, 10)

    # Ensure the features are in the right shape
    epilines_left = cv2.computeCorrespondEpilines(features_right.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    epilines_right = cv2.computeCorrespondEpilines(features_left.reshape(-1, 1, 2), 1, F).reshape(-1, 3)

    # Draw epipolar lines
    left_with_epilines = draw_epipolar_lines(left_image, epilines_left, features_left)
    right_with_epilines = draw_epipolar_lines(right_image, epilines_right, features_right)

    # Display epipolar lines
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(left_with_epilines, cv2.COLOR_BGR2RGB))
    plt.title('Left Image with Epilines')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(right_with_epilines, cv2.COLOR_BGR2RGB))
    plt.title('Right Image with Epilines')
    plt.show()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test 3 - Rectify calibration_images 02-pair and draw horizontal lines to check corresponding points lie the same line
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_3():

    # Load calibration parameters
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_params_path = os.path.join(curr_dir, 'calibration_params.npz')
    calibration_params = np.load(calibration_params_path)

    K1 = calibration_params['K1']
    D1 = calibration_params['D1']
    K2 = calibration_params['K2']
    D2 = calibration_params['D2']
    R = calibration_params['R']
    T = calibration_params['T']

    # Load left and right images
    left_img_path = os.path.join(curr_dir, 'calibration_images/left_02.jpg')
    right_img_path = os.path.join(curr_dir, 'calibration_images/right_02.jpg')

    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)

    if img_left is None or img_right is None:
        print("Error: Unable to load images.")
        exit()

    # Convert images to grayscale
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # Get image dimensions
    h, w = gray_left.shape

    # Stereo rectification
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        K1, D1, K2, D2, (w, h), R, T, alpha=0
    )

    # Create rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (w, h), cv2.CV_32FC1)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (w, h), cv2.CV_32FC1)

    # Rectify images
    rectified_left = cv2.remap(img_left, left_map1, left_map2, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(img_right, right_map1, right_map2, cv2.INTER_LINEAR)

    # Draw horizontal lines on rectified images for validation
    def draw_horizontal_lines(img, line_spacing=50):
        for y in range(0, img.shape[0], line_spacing):
            cv2.line(img, (0, y), (img.shape[1], y), (0, 255, 0), 1)

    rectified_left_with_lines = rectified_left.copy()
    rectified_right_with_lines = rectified_right.copy()
    draw_horizontal_lines(rectified_left_with_lines)
    draw_horizontal_lines(rectified_right_with_lines)

    # Display rectified images
    cv2.imshow("Rectified Left", rectified_left_with_lines)
    cv2.imshow("Rectified Right", rectified_right_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Save rectified images (optional)
    # cv2.imwrite("rectified_left.jpg", rectified_left)
    # cv2.imwrite("rectified_right.jpg", rectified_right)

    print("Stereo rectification completed.")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Test 4 - Compute a disparity map with cv2.StereoSGBM_create() and validate the resulting depth map
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def test_4():

    # Load the stereo calibration parameters
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    calibration_params_path = os.path.join(curr_dir, 'calibration_params.npz')
    calibration_params = np.load(calibration_params_path)
    K1, D1, K2, D2 = calibration_params['K1'], calibration_params['D1'], calibration_params['K2'], calibration_params['D2']
    R, T = calibration_params['R'], calibration_params['T']

    # Load a pair of stereo images
    left_image_path = os.path.join(curr_dir, 'calibration_validation/left_depth_testing.jpg')
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image_path = os.path.join(curr_dir, 'calibration_validation/right_depth_testing.jpg')
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    # Get image dimensions
    image_size = left_image.shape[::-1][1:]

    # Stereo rectification (maps for undistortion and rectification)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T)

    map1_left, map2_left = cv2.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv2.CV_32FC1)

    # Rectify the images
    rectified_left = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
    rectified_right = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)

    # Compute disparity map using StereoSGBM
    window_size = 5
    min_disp = 0
    num_disp = 16 * 5  # Must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=15,           # blockSize=21 makes a difference
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,     # uniquenessRatio=20 makes a difference
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

    # Normalize disparity map for visualization
    disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Display results
    cv2.imshow("Rectified Left Image", rectified_left)
    cv2.imshow("Rectified Right Image", rectified_right)
    cv2.imshow("Disparity Map", disparity_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save disparity map (optional)
    cv2.imwrite('calibration_validation/disparity_map.jpg', disparity_vis)

    plt.imshow(disparity, 'gray')
    plt.title('Disparity Map')
    plt.colorbar()
    plt.show()

    # Compute depth map using Q matrix
    depth_map = cv2.reprojectImageTo3D(disparity, Q)

    # Save depth map (optional)
    cv2.imwrite('calibration_validation/depth_map.jpg', depth_map)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def caputre_image_pair(pair_name):

    # Set up save directory if it doesn't exist
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(curr_dir, 'calibration_validation')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Open the camera and set resolution & fps
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap.set(cv2.CAP_PROP_FPS, 120)

    if not cap.isOpened():
        print("Error: Could not open stereo camera.")
        return

    print("Press 's' to save an R-L image pair or 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stereo camera.")
            break
        
        # Split the stereo frame into left and right images
        _, width, _ = frame.shape
        frame_left = frame[:, :width // 2]
        frame_right = frame[:, width // 2:]

        # Display the left and right images
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'): # save image
            left_image_path = os.path.join(save_dir, f"left_{pair_name}.jpg")
            right_image_path = os.path.join(save_dir, f"right_{pair_name}.jpg")
            cv2.imwrite(left_image_path, frame_left)
            cv2.imwrite(right_image_path, frame_right)
            print(f"Captured stereo pair and saved to {save_dir}/ directory")
            break
        elif key == ord('q'): # quit
            print("Terminated without capturing an image pair.")
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Choose which calibration test to execute 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

run_test = 4

if run_test == 1:
    test_1()
elif run_test == 2:
    test_2()
elif run_test == 3:
    test_3()
elif run_test == 4:
    test_4()
elif run_test == 5:
    caputre_image_pair('depth_testing')
else:
    print("Please choose which test you would like to run by adjusting run_test variable")