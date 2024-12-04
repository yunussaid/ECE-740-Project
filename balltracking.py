import cv2
import numpy as np
from scipy.optimize import curve_fit
import time


# Load calibration data
calibration_data = np.load('stereo_params.npz')
focal_length = calibration_data['focal_length']  # Focal length in pixels
baseline = calibration_data['baseline']  # Baseline in cm
principal_point = calibration_data['principal_point']  # (cx, cy)

# Use these parameters in your 3D position calculation
stereo_params = {
    'focal_length': focal_length,
    'baseline': baseline,
    'principal_point': principal_point
}

# # Stereo camera parameters (replace with your calibration values)
# stereo_params = {
#     'focal_length': 1280,
#     'baseline': 0.06       # Distance between the cameras in centimeters
# }

# Global variables for tracking
timestamps = []
positions = []

def calculate_3d_position(point_left, point_right, stereo_params, image_width, image_height):
    """
    Calculate the 3D position of the ball in centimeters using stereo vision.

    Args:
        point_left (tuple): (x, y) coordinates of the ball in the left image.
        point_right (tuple): (x, y) coordinates of the ball in the right image.
        stereo_params (dict): Contains 'focal_length', 'baseline', and 'principal_point'.
        image_width (int): Width of the image (in pixels).
        image_height (int): Height of the image (in pixels).

    Returns:
        tuple: (X, Y, Z) 3D coordinates of the ball in centimeters.
    """
    disparity = point_left[0] - point_right[0]
    if disparity <= 0:  # Avoid division by zero or negative disparity
        return None

    focal_length = stereo_params['focal_length']
    baseline = stereo_params['baseline']  # Baseline in cm
    principal_point = stereo_params['principal_point']  # (cx, cy)

    # Calculate depth (Z) in cm
    Z = (focal_length * baseline) / disparity

    # Adjust the origin to the center of the stereo camera
    cx, cy = principal_point
    X = ((point_left[0] - cx) * Z) / focal_length - (baseline / 2)
    Y = -((point_left[1] - cy) * Z) / focal_length

    return X, Y, Z


def trajectory_model(t, x0, y0, z0, vx, vy, vz, az):
    """Model the trajectory of the ball."""
    x = x0 + vx * t
    y = y0 + vy * t
    z = z0 + vz * t + 0.5 * az * t**2
    return np.stack([x, y, z], axis=-1)

def fit_trajectory(timestamps, positions):
    """Fit the observed trajectory to a parabolic motion model."""
    x_data, y_data, z_data = positions.T
    initial_guess = [x_data[0], y_data[0], z_data[0], 0, 0, 0, -9.8]
    popt, _ = curve_fit(
        lambda t, x0, y0, z0, vx, vy, vz, az: trajectory_model(t, x0, y0, z0, vx, vy, vz, az)[:, 2],
        timestamps,
        z_data,
        p0=initial_guess
    )
    return popt

def orangeContour(frame):
    """
    Detect the orange ping-pong ball in the given frame and return its center, radius, and mask.

    Args:
        frame (numpy.ndarray): Input frame from the camera.

    Returns:
        tuple: (center, radius, mask)
            - center: (x, y) coordinates of the ball's center.
            - radius: Radius of the detected ball.
            - mask: Binary mask highlighting the detected ball.
    """
    # Define the color range for the orange ping-pong ball in HSV
    lower_orange = np.array([10, 150, 150])  # Lower bound for orange
    upper_orange = np.array([25, 255, 255])  # Upper bound for orange

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask to only keep the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours of the masked image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out small contours by area
        if cv2.contourArea(contour) > 3:  # Adjust threshold based on the size of the ball
            # Draw a bounding circle around the detected ball
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Only proceed if the radius meets a minimum size
            if radius > 0.5:
                return center, radius, mask

    # Return None if no ball is detected
    return None, None, mask

# Translation vector: camera to arm (in cm)
translation_vector = (0, 455, 395)  # (T_x, T_y, T_z)


def transform_to_arm_coordinates(camera_position, translation_vector):
    """
    Transform the ball's 3D position from the camera's coordinate system to the robotic arm's coordinate system.

    Args:
        camera_position (tuple): (X_c, Y_c, Z_c) position of the ball in the camera's coordinates.
        translation_vector (tuple): (T_x, T_y, T_z) translation from the camera to the arm.

    Returns:
        tuple: (X_a, Y_a, Z_a) position of the ball in the arm's coordinates.
    """
    X_c, Y_c, Z_c = camera_position
    T_x, T_y, T_z = translation_vector

    # Apply the translation matrix
    X_a = X_c - T_x
    Y_a = Y_c + T_y
    Z_a = Z_c - T_z

    return X_a, Y_a, Z_a

def balltracking():
    """Main function to integrate ball detection, tracking, and prediction."""
    # Open the stereo camera (single index for the stereo stream)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from the camera.")
            break

        # Assuming the stereo camera outputs a side-by-side (SBS) image
        height, width, _ = frame.shape
        half_width = width // 2

        # Split the frame into left and right images
        left_frame = frame[:, :half_width]
        right_frame = frame[:, half_width:]

        # Detect ball in left and right frames
        ball_left,_, mask_left = orangeContour(left_frame)  # Assume it returns ball center
        ball_right,_, mask_right = orangeContour(right_frame)

        if ball_left and ball_right:
            # Calculate the ball's 3D position
            pos_3d_camera = calculate_3d_position(
                    ball_left, ball_right, stereo_params, width, height
                )
            # print("Ball Position in Camera Coordinates:", pos_3d_camera)
            if pos_3d_camera != None:
                pos_3d_arm = transform_to_arm_coordinates(pos_3d_camera, translation_vector)
                # print("Ball Position in Arm Coordinates:", pos_3d_arm)

                if pos_3d_arm != None:
                    x, z, y = pos_3d_arm
                    if x != None and y != None and z != None:
                        print(f"Ball position: (x:{x:.2f},\ty: {y:.2f},\tz: {z:.2f})")

            # if pos_3d is not None:
            #     timestamps.append(time.time())
            #     positions.append(pos_3d)

            #     # Fit trajectory after collecting enough data points
            #     if len(positions) > 6:
            #         positions_np = np.array(positions)
            #         timestamps_np = np.array(timestamps) - timestamps[0]
            #         trajectory_params = fit_trajectory(timestamps_np, positions_np)

            #         # Predict future position
            #         t_future = 1.5  # Predict 1.5 seconds ahead
            #         future_position = trajectory_model(
            #             np.array([t_future]), *trajectory_params
            #         )
            #         print("Predicted Future Position:", future_position)

            #     # Manage buffer size
            #     if len(positions) > 20:
            #         positions.pop(0)
            #         timestamps.pop(0)

        # Show frames and masks for debugging
        cv2.imshow("Left Camera", left_frame)
        cv2.imshow("Right Camera", right_frame)
        if ball_left:
            cv2.imshow("Left Mask", mask_left)
        if ball_right:
            cv2.imshow("Right Mask", mask_right)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    balltracking()
