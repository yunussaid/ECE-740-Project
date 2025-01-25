import cv2
import numpy as np

class Camera:
    def __init__(self, show_frames=False, show_masks=False, dynamic_mode=False):
        self.show_frames = show_frames
        self.show_masks = show_masks
        self.dynamic_mode = dynamic_mode

        # Load calibration parameters
        calibration_data = np.load('calibration_params.npz')
        self.K1, self.D1 = calibration_data['K1'], calibration_data['D1']
        self.K2, self.D2 = calibration_data['K2'], calibration_data['D2']
        self.R, self.T = calibration_data['R'], calibration_data['T']
        self.baseline = calibration_data['baseline']

        # Define default HSV bounds and minimum radius for detection
        self.DEFAULT_LOWER_COLOR = np.array([0, 100, 165])
        self.DEFAULT_UPPER_COLOR = np.array([20, 255, 255])
        self.DEFAULT_MIN_RADIUS = 1

        # Initialize dynamic trackbars if needed
        if self.dynamic_mode:
            cv2.namedWindow("Trackbars")
            cv2.createTrackbar("Lower H", "Trackbars", 0, 179, lambda x: None)
            cv2.createTrackbar("Lower S", "Trackbars", 100, 255, lambda x: None)
            cv2.createTrackbar("Lower V", "Trackbars", 165, 255, lambda x: None)
            cv2.createTrackbar("Upper H", "Trackbars", 20, 179, lambda x: None)
            cv2.createTrackbar("Upper S", "Trackbars", 255, 255, lambda x: None)
            cv2.createTrackbar("Upper V", "Trackbars", 255, 255, lambda x: None)
            cv2.createTrackbar("Min Radius", "Trackbars", 1, 20, lambda x: None) # Scaled by 10x

        # Initialize camera feed
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) # can use cv2.CAP_DSHOW strictly for faster testing
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.cap.set(cv2.CAP_PROP_FPS, 120)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open stereo camera.")

    def preprocess(self, frame):
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        return hsv

    def detect_ball(self, frame):
        if self.dynamic_mode:
            lower_h = cv2.getTrackbarPos("Lower H", "Trackbars")
            lower_s = cv2.getTrackbarPos("Lower S", "Trackbars")
            lower_v = cv2.getTrackbarPos("Lower V", "Trackbars")
            upper_h = cv2.getTrackbarPos("Upper H", "Trackbars")
            upper_s = cv2.getTrackbarPos("Upper S", "Trackbars")
            upper_v = cv2.getTrackbarPos("Upper V", "Trackbars")
            min_radius = cv2.getTrackbarPos("Min Radius", "Trackbars") / 10.0  # Scale back to float

            lower_color = np.array([lower_h, lower_s, lower_v])
            upper_color = np.array([upper_h, upper_s, upper_v])
        else:
            lower_color = self.DEFAULT_LOWER_COLOR
            upper_color = self.DEFAULT_UPPER_COLOR
            min_radius = self.DEFAULT_MIN_RADIUS

        # Apply color filter and preprocess the resultant mask
        mask = cv2.inRange(frame, lower_color, upper_color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
            if radius >= min_radius:
                return (int(x), int(y), int(radius)), mask

        return None, mask

    def triangulate_points(self, x_left, y_left, x_right, y_right):
        P1 = np.dot(self.K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
        P2 = np.dot(self.K2, np.hstack((self.R, self.T)))

        points_left = np.array([[x_left, y_left]], dtype=np.float64).T
        points_right = np.array([[x_right, y_right]], dtype=np.float64).T

        points_4d = cv2.triangulatePoints(P1, P2, points_left, points_right)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.flatten()

    def get_ball_position(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        _, width, _ = frame.shape
        frame_left = frame[:, :width // 2]
        frame_right = frame[:, width // 2:]

        hsv_left = self.preprocess(frame_left)
        hsv_right = self.preprocess(frame_right)

        circle_left, mask_left = self.detect_ball(hsv_left)
        circle_right, mask_right = self.detect_ball(hsv_right)

        if self.show_masks:
            cv2.imshow("Masked Left", mask_left)
            cv2.imshow("Masked Right", mask_right)

        ball_position = None
        
        if circle_left is not None and circle_right is not None:
            x_left, y_left, radius_left = circle_left
            x_right, y_right, radius_right = circle_right

            point_3d = self.triangulate_points(x_left, y_left, x_right, y_right)
            point_3d[0] -= self.baseline / 2
            point_3d[1] *= -1
            ball_position = [int(coord) for coord in point_3d]

            cv2.circle(frame_left, (x_left, y_left), int(radius_left), (0, 255, 0), 2)
            cv2.circle(frame_right, (x_right, y_right), int(radius_right), (0, 255, 0), 2)
            cv2.putText(frame_left, f"3D: [{''.join(f'{int(coord):>6}' for coord in point_3d)}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame_right, f"3D: [{''.join(f'{int(coord):>6}' for coord in point_3d)}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if self.show_frames:
            cv2.imshow("Left Camera", frame_left)
            cv2.imshow("Right Camera", frame_right)

        return ball_position

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = Camera(show_frames=True, show_masks=True, dynamic_mode=False)

    while True:
        ball_position = camera.get_ball_position()
        if ball_position:
            print(f"Ball Position: [{''.join(f'{coord:>6}' for coord in ball_position)}]")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
