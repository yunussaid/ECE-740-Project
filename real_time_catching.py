import time
import numpy as np
from arm import Arm
from scipy.optimize import curve_fit
import cv2
from balltracking import orangeContour, calculate_3d_position,fit_trajectory,trajectory_model

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

# Global variables for tracking
timestamps = []
positions = []

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

def main():
    # Initialize the robotic arm
    arm = Arm()

    try:
        # Connect to the robotic arm
        arm.connect()
        print("Robotic arm connected.")

        # Open the stereo camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open the camera.")
            return

        while True:
            # Capture a frame from the stereo camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from the camera.")
                break

            # Split the stereo frame into left and right images
            height, width, _ = frame.shape
            half_width = width // 2
            left_frame = frame[:, :half_width]
            right_frame = frame[:, half_width:]

            # Detect the ball in the left and right frames
            ball_left, _, mask_left = orangeContour(left_frame)
            ball_right, _, mask_right = orangeContour(right_frame)

            if ball_left and ball_right:
                # Calculate the ball's 3D position in camera coordinates
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

                if pos_3d_camera is not None:
                    # Append current position and timestamp
                    timestamps.append(time.time())
                    positions.append(pos_3d_camera)

                    # Fit trajectory after collecting enough data points
                    if len(positions) > 6:
                        positions_np = np.array(positions)
                        timestamps_np = np.array(timestamps) - timestamps[0]
                        trajectory_params = fit_trajectory(timestamps_np, positions_np)

                        # Predict the future position in arm coordinates
                        t_future = 1.0  # Predict 1 second ahead
                        future_position_camera = trajectory_model(
                            np.array([t_future]), *trajectory_params
                        )[0]

                        # Move the robotic arm to the predicted position
                        # Transform to arm coordinates
                        future_position_arm = transform_to_arm_coordinates(future_position_camera, translation_vector)
                        x, y, z = future_position_arm
                        try:
                            arm.move_net_to_xy(x, y, True, True)  # z, which is the depth of the camera, is actually y axis of the arm
                            print(f"Arm moved to position: X={x}, Y={y}")
                        except ValueError as e:
                            print(f"Arm movement error: {e}")

                    # Manage the buffer size
                    if len(positions) > 20:
                        positions.pop(0)
                        timestamps.pop(0)

            # Show debug frames
            cv2.imshow("Left Camera", left_frame)
            cv2.imshow("Right Camera", right_frame)
            if ball_left:
                cv2.imshow("Left Mask", mask_left)
            if ball_right:
                cv2.imshow("Right Mask", mask_right)

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print("Error occurred:", e)

    finally:
        # Clean up resources
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        arm.disconnect()
        print("Program exited and resources released.")

if __name__ == "__main__":
    main()
