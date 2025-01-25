import cv2
import numpy as np

# Load calibration parameters
calibration_data = np.load('calibration_params.npz')
K1, D1 = calibration_data['K1'], calibration_data['D1']
K2, D2 = calibration_data['K2'], calibration_data['D2']
R, T = calibration_data['R'], calibration_data['T']
baseline = calibration_data['baseline']

# Define default HSV bounds
DEFAULT_LOWER_COLOR = np.array([0, 100, 165])
DEFAULT_UPPER_COLOR = np.array([20, 255, 255])

# Define minimum ball radius for detection
MIN_DETECTABLE_RADIUS = 1

# Function to adjust HSV bounds dynamically
def nothing(x):
    pass

# Option to use dynamic adjustment
use_dynamic_hsv = False  # Change to True to enable trackbars dynamically

if use_dynamic_hsv:
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("Lower H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("Lower S", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("Lower V", "Trackbars", 100, 255, nothing)
    cv2.createTrackbar("Upper H", "Trackbars", 40, 179, nothing)
    cv2.createTrackbar("Upper S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("Upper V", "Trackbars", 255, 255, nothing)

# Preprocessing function
def preprocess(frame):
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv

# Ball detection using color and contours
def detect_ball(frame):
    if use_dynamic_hsv:
        # Get dynamic bounds from trackbars
        lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
        lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
        lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
        upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
        upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
        upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")

        lower_color = np.array([lower_h, lower_s, lower_v])
        upper_color = np.array([upper_h, upper_s, upper_v])
    else:
        lower_color = DEFAULT_LOWER_COLOR
        upper_color = DEFAULT_UPPER_COLOR

    # Apply color filter
    mask = cv2.inRange(frame, lower_color, upper_color)

    # Mask preprocessing
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the ball
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Ensure the detected object is large enough to be the ball
        if radius >= MIN_DETECTABLE_RADIUS:
            return (int(x), int(y), int(radius)), mask

    return None, mask

# Triangulation function
def triangulate_points(x_left, y_left, x_right, y_right, K1, K2, R, T):
    # Prepare projection matrices
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K2, np.hstack((R, T)))

    # Homogeneous 2D points
    points_left = np.array([[x_left, y_left]], dtype=np.float64).T
    points_right = np.array([[x_right, y_right]], dtype=np.float64).T

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, points_left, points_right)
    points_3d = points_4d[:3] / points_4d[3]  # Convert to non-homogeneous
    return points_3d.flatten()

# Set up stereo camera feed (resolution & fps included)
cap = cv2.VideoCapture(1) # can use cv2.CAP_DSHOW for faster testing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 120)

if not cap.isOpened():
    print("Error: Could not open stereo camera.")
    exit()

while True:
    # Capture stereo frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from stereo camera.")
        break

    # Split the stereo frame into left and right images
    height, width, _ = frame.shape
    frame_left = frame[:, :width // 2]
    frame_right = frame[:, width // 2:]

    # Preprocess frames
    hsv_left = preprocess(frame_left)
    hsv_right = preprocess(frame_right)

    # Detect the ball in both frames
    circle_left, mask_left = detect_ball(hsv_left)
    circle_right, mask_right = detect_ball(hsv_right)

    # Display preprocessed and masked frames
    cv2.imshow("Masked Left", mask_left)
    cv2.imshow("Masked Right", mask_right)

    if circle_left is not None and circle_right is not None:
        # Use the detected circle
        x_left, y_left, radius_left = circle_left
        x_right, y_right, radius_right = circle_right

        # Triangulate to find 3D coordinates
        point_3d = triangulate_points(x_left, y_left, x_right, y_right, K1, K2, R, T)

        # Adjust coordinates to make the middle of the baseline (0, 0, 0)
        point_3d[0] -= baseline / 2  # Adjust X-coordinate

        # Adjust coordinates to make upwards positive Y
        point_3d[1] *= -1  # Adjust Y-coordinate

        print(f"3D Coordinates: [{''.join(f'{int(coord):>6}' for coord in point_3d)}]")

        # Draw circles and coordinates on frames
        cv2.circle(frame_left, (int(x_left), int(y_left)), int(radius_left), (0, 255, 0), 2)
        cv2.circle(frame_right, (int(x_right), int(y_right)), int(radius_right), (0, 255, 0), 2)
        cv2.putText(frame_left, f"3D: [{''.join(f'{int(coord):>6}' for coord in point_3d)}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    # Display frames
    cv2.imshow("Left Camera", frame_left)
    cv2.imshow("Right Camera", frame_right)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
