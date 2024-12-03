import cv2
import numpy as np
from scipy.optimize import curve_fit
import time

# Stereo camera parameters (replace with your calibration values)
stereo_params = {
    'focal_length': 1000,
    'baseline': 0.06       # Distance between the cameras in centimeters
}

# Global variables for tracking
timestamps = []
positions = []

def calculate_3d_position(point_left, point_right, stereo_params, image_width, image_height):
    """
    Calculate the 3D position of the ball using stereo vision, with respect to the center of the image.

    Args:
        point_left (tuple): (x, y) coordinates of the ball in the left image.
        point_right (tuple): (x, y) coordinates of the ball in the right image.
        stereo_params (dict): Contains 'focal_length' and 'baseline'.
        image_width (int): Width of the image (in pixels).
        image_height (int): Height of the image (in pixels).

    Returns:
        tuple: (X, Y, Z) 3D coordinates of the ball.
    """
    disparity = point_left[0] - point_right[0]
    if disparity == 0:  # Avoid division by zero
        return None

    focal_length = stereo_params['focal_length']
    baseline = stereo_params['baseline']
    depth = (focal_length * baseline) / disparity

    # Adjust the origin to the center of the image
    center_x = image_width / 2
    center_y = image_height / 2

    # Calculate 3D coordinates with the center of the image as the origin
    X = ((point_left[0] - center_x) * depth) / focal_length 
    Y = ((point_left[1] - center_y) * depth) / focal_length
    Z = depth
    X = X
    Y = -Y*100
    Z = Z*20

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
    lower_orange = np.array([5, 150, 150])  # Lower bound for orange
    upper_orange = np.array([15, 255, 255])  # Upper bound for orange

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

def main():
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
            pos_3d = calculate_3d_position(ball_left, ball_right, stereo_params, width, height)
            print(pos_3d)
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
    main()