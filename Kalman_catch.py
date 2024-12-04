from filterpy.kalman import KalmanFilter
import time
import numpy as np
from arm import Arm
from scipy.optimize import curve_fit
import cv2
from balltracking import orangeContour, calculate_3d_position, fit_trajectory, trajectory_model

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

# Translation vector: camera to arm (in cm)
translation_vector = (0, 455, 395)  # (T_x, T_y, T_z)

# Global variables for tracking
timestamps = []
positions = []
last_position = None  # Track last arm position

# Initialize Kalman Filter
def setup_kalman_filter():
    dt = 0.033  # Time step (33 ms for ~30 FPS)
    kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [X, Y, Z, Vx, Vy, Vz], Measurement: [X, Y, Z]
    
    # State transition matrix (F)
    kf.F = np.array([
        [1, 0, 0, dt, 0,  0],  # X = X + Vx * dt
        [0, 1, 0, 0,  dt, 0],  # Y = Y + Vy * dt
        [0, 0, 1, 0,  0,  dt], # Z = Z + Vz * dt
        [0, 0, 0, 1,  0,  0],  # Vx = Vx
        [0, 0, 0, 0,  1,  0],  # Vy = Vy
        [0, 0, 0, 0,  0,  1]   # Vz = Vz
    ])
    
    # Measurement function (H)
    kf.H = np.array([
        [1, 0, 0, 0, 0, 0],  # Measure X
        [0, 1, 0, 0, 0, 0],  # Measure Y
        [0, 0, 1, 0, 0, 0]   # Measure Z
    ])
    
    # Measurement noise covariance (R): Adjust based on the camera's accuracy
    kf.R = np.eye(3) * 5  # Small noise for accurate sensors
    
    # Process noise covariance (Q)
    kf.Q = np.eye(6) * 0.1  # Adjust for smoother or more responsive predictions
    
    # Initial state covariance (P)
    kf.P = np.eye(6) * 500  # Initial uncertainty for position and velocity
    
    # Initial state vector
    kf.x = np.zeros(6)  # [X, Y, Z, Vx, Vy, Vz]
    
    return kf

# Apply translation to arm coordinates
def transform_to_arm_coordinates(camera_position, translation_vector):
    X_c, Y_c, Z_c = camera_position
    T_x, T_y, T_z = translation_vector

    # Apply the translation matrix
    X_a = X_c - T_x
    Y_a = Y_c + T_y
    Z_a = Z_c - T_z

    return X_a, Y_a, Z_a

# Check if the arm needs to move to the new position
def should_update_position(current_position, new_position, threshold=5.0):
    if current_position is None:
        return True  # Always update if no previous position exists

    diff = np.linalg.norm(np.array(current_position) - np.array(new_position))
    return diff > threshold

# Interpolate between current and target position for smoother motion
def interpolate_position(current_position, target_position, steps=10):
    current = np.array(current_position)
    target = np.array(target_position)
    return [tuple(current + (target - current) * i / steps) for i in range(1, steps + 1)]

def main():
    # Initialize the robotic arm
    arm = Arm()
    kf = setup_kalman_filter()  # Initialize Kalman Filter

    try:
        # Connect to the robotic arm
        arm.connect()
        print("Robotic arm connected.")

        # Open the stereo camera
        cap = cv2.VideoCapture(0)

        # Lower the camera resolution to increase the frame rate and reduce processing time.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

                if pos_3d_camera is not None:
                    # Predict the next state
                    kf.predict()

                    # Update the Kalman filter with the measured position
                    measured_position = np.array(pos_3d_camera[:3])  # Only (X, Y, Z)
                    kf.update(measured_position)

                    # Get the filtered position
                    filtered_position_camera = kf.x[:3]  # [X, Y, Z]
                    filtered_position_arm = transform_to_arm_coordinates(filtered_position_camera, translation_vector)
                    print(f"Filtered Ball Position (Arm Coordinates): {filtered_position_arm}")

                    # Predict the future position
                    if len(positions) > 6:
                        positions_np = np.array(positions)
                        timestamps_np = np.array(timestamps) - timestamps[0]
                        trajectory_params = fit_trajectory(timestamps_np, positions_np)

                        # Predict the future position
                        t_future = 1.5  # Predict farther ahead for better reaction
                        future_position_camera = trajectory_model(
                            np.array([t_future]), *trajectory_params
                        )[0]
                        future_position_arm = transform_to_arm_coordinates(future_position_camera, translation_vector)

                        # Move the arm only if needed
                        if should_update_position(last_position, future_position_arm, threshold=5.0):
                            # Interpolate for smoother movement
                            if last_position is not None:
                                interpolated_positions = interpolate_position(last_position, future_position_arm, steps=5)
                                for pos in interpolated_positions:
                                    x, y, z = pos
                                    arm.move_net_to_xy(x, y, True, True)
                                    time.sleep(0.05)  # Small delay for smooth interpolation
                            else:
                                x, y, z = future_position_arm
                                arm.move_net_to_xy(x, y, True, True)

                            last_position = future_position_arm  # Update last position

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
