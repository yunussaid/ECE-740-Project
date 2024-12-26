import cv2
import os

def capture_stereo_images(output_dir, num_images=20):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the camera and set resolution & fps
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    # cv2.CAP_DSHOW is fast during setup but lags during runtime. It's used here because lag
    # doesn't really matter for capturing callibaration images.

    if not cap.isOpened():
        print("Error: Could not open stereo camera.")
        return

    print(f"Starting capture. {num_images} stereo pairs will be saved to '{output_dir}'...")

    captured_images = 0

    while captured_images < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from stereo camera.")
            break

        # Split the stereo frame into left and right images
        height, width, _ = frame.shape
        frame_left = frame[:, :width // 2]
        frame_right = frame[:, width // 2:]

        # Display the left and right images
        cv2.imshow("Left Camera", frame_left)
        cv2.imshow("Right Camera", frame_right)

        key = cv2.waitKey(5) & 0xFF
        
        if key == ord('s'): # save image
            left_image_path = os.path.join(output_dir, f"left_{captured_images + 1:02d}.jpg")
            right_image_path = os.path.join(output_dir, f"right_{captured_images + 1:02d}.jpg")
            cv2.imwrite(left_image_path, frame_left)
            cv2.imwrite(right_image_path, frame_right)
            print(f"Captured stereo pair {captured_images + 1}/{num_images}")
            captured_images += 1
        elif key == ord('q'): # quit
            print("Capture terminated early by user.")
            break

    print("Capture complete. All images saved.")
    cap.release()
    cv2.destroyAllWindows()


def main():
    capture_stereo_images("calibration_images", 20)


if __name__ == "__main__":
    main()