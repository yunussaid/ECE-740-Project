from filterpy.kalman import KalmanFilter
import time
import numpy as np
from arm import Arm
import cv2
from balltracking import orangeContour, calculate_3d_position

# Load calibration data
calibration_data = np.load('stereo_params.npz')
focal_length = calibration_data['focal_length']  # Focal length in pixels
baseline = calibration_data['baseline']  # Baseline in cm
principal_point = calibration_data['principal_point']  # (cx, cy)

stereo_params = {
    'focal_length': focal_length,
    'baseline': baseline,
    'principal_point': principal_point
}

# Translation vector: camera to arm (in cm)
translation_vector = (0, 455, 395)  # (T_x, T_y, T_z)


def transform_to_arm_coordinates(camera_position, translation_vector):
    """Transform 3D position from camera to arm coordinates."""
    X_c, Y_c, Z_c = camera_position
    T_x, T_y, T_z = translation_vector
    X_a = X_c - T_x
    Y_a = Y_c + T_y
    Z_a = Z_c - T_z
    return X_a, Y_a, Z_a


def setup_kalman_filter():
    """Initialize the Kalman Filter."""
    dt = 0.033  # Time step (~30 FPS)
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

    # Measurement noise covariance (R)
    kf.R = np.eye(3) * 5  # Adjust based on camera accuracy

    # Process noise covariance (Q)
    kf.Q = np.eye(6) * 0.1  # Adjust for smoother or more responsive predictions

    # Initial state covariance (P)
    kf.P = np.eye(6) * 500

    # Initial state vector
    kf.x = np.zeros(6)  # [X, Y, Z, Vx, Vy, Vz]

    return kf


def main():
    # Initialize the robotic arm
    arm = Arm()
    kf = setup_kalman_filter()  # Initialize Kalman Filter

    try:
        arm.connect()
        print("Robotic arm connected.")

        # Open stereo camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open the camera.")
            return

        while True:
            # Capture frame from stereo camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from the camera.")
                break

            # Split the stereo frame into left and right images
            height, width, _ = frame.shape
            half_width = width // 2
            left_frame = frame[:, :half_width]
            right_frame = frame[:, half_width:]

            # Detect ball in left and right frames
            ball_left, _, mask_left = orangeContour(left_frame)
            ball_right, _, mask_right = orangeContour(right_frame)

            if ball_left and ball_right:
                # Calculate 3D position in camera coordinates
                pos_3d_camera = calculate_3d_position(ball_left, ball_right, stereo_params, width, height)
                if pos_3d_camera is not None:
                    # Predict the next state
                    kf.predict()

                    # Update Kalman Filter with the measured position
                    measured_position = np.array(pos_3d_camera[:3])  # Only (X, Y, Z)
                    kf.update(measured_position)

                    # Get filtered position
                    filtered_position_camera = kf.x[:3]
                    filtered_position_arm = transform_to_arm_coordinates(filtered_position_camera, translation_vector)
                    print(f"Filtered Position (Arm Coordinates): {filtered_position_arm}")

                    # Predict future position
                    # Calculate dynamic t_future
                    distance_to_arm = np.linalg.norm(filtered_position_arm)  # Distance from ball to arm
                    average_velocity = np.linalg.norm(kf.x[3:])  # Magnitude of velocity vector
                    t_future = max(0.5, min(2.0, distance_to_arm / average_velocity))  # Adjust range as needed

                    # t_future = 2  # Time to predict into the future (seconds)
                    future_position = filtered_position_camera + kf.x[3:] * t_future  # X + Vx*t, Y + Vy*t, Z + Vz*t
                    future_position_arm = transform_to_arm_coordinates(future_position, translation_vector)
                    print(f"Predicted Future Position (Arm Coordinates): {future_position_arm}")

                    # Move the arm to the predicted position
                    x, y, z = future_position_arm
                    try:
                        arm.move_net_to_xy(x, y, True, True)  # Adjust axes if necessary
                        print(f"Arm moved to position: X={x:.2f}, Y={y:.2f}")
                    except ValueError as e:
                        print(f"Arm movement error: {e}")

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
        # Release resources
        if cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

        arm.disconnect()
        print("Program exited and resources released.")


if __name__ == "__main__":
    main()
