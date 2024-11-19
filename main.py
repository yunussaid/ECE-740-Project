import cv2
import numpy as np


def filterOrange():
    # Open the camera
    cap = cv2.VideoCapture(1)
    
    # Define the color range for the orange Ping-Pong ball in HSV
    # These values may need slight tweaking depending on your lighting conditions
    lower_orange = np.array([5, 150, 150])  # Lower bound for orange
    upper_orange = np.array([15, 255, 255])  # Upper bound for orange

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask for the Ping-Pong ball
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Filter out everything but the ball
        result = cv2.bitwise_and(frame, frame, mask=mask)
        
        cv2.imshow("Masked Image", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


def orangeContour():
    # Open the camera
    cap = cv2.VideoCapture(1)

    # Define the color range for the orange Ping-Pong ball in HSV
    # These values may need slight tweaking depending on your lighting conditions
    lower_orange = np.array([5, 150, 150])  # Lower bound for orange
    upper_orange = np.array([15, 255, 255])  # Upper bound for orange

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image from BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create a mask to only keep the orange color
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Find contours of the masked image
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter out small contours by area
            if cv2.contourArea(contour) > 100:  # Adjust threshold based on the size of the ball
                # Draw a bounding circle around the detected ball
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Only proceed if the radius meets a minimum size
                if radius > 10:
                    cv2.circle(frame, center, radius, (0, 255, 0), 2)  # Draw green circle around the ball
                    cv2.putText(frame, "Ping-Pong Ball", (center[0] - 10, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the original frame with contours drawn on it
        cv2.imshow("Detected Ping-Pong Ball", frame)
        
        # Display the mask (for debugging purposes)
        cv2.imshow("Mask", mask)
        
        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


def main():
    # filterOrange()
    orangeContour()

if __name__ == "__main__":
    main()