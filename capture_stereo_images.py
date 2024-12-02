# import cv2
# import os

# def capture_stereo_images(output_dir, num_images=20, delay=1):
#     """
#     Capture stereo images for calibration.
    
#     :param output_dir: Directory to save captured images.
#     :param num_images: Number of stereo pairs to capture.
#     :param delay: Delay (in seconds) between captures.
#     """
#     # Create output directory if it doesn't exist
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     print("Created output_dir:", output_dir)

#     # Open the stereo camera
#     cap = cv2.VideoCapture(1)  # Adjust if necessary for your stereo camera index

#     if not cap.isOpened():
#         print("Error: Could not open stereo camera.")
#         return

#     print("Press 's' to start capturing stereo images.")
#     print("Press 'q' to quit without capturing.")

#     captured_images = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame from stereo camera.")
#             break

#         # Split the stereo frame into left and right images
#         height, width, _ = frame.shape
#         frame_left = frame[:, :width // 2]
#         frame_right = frame[:, width // 2:]

#         # Display the left and right images
#         cv2.imshow("Left Camera", frame_left)
#         cv2.imshow("Right Camera", frame_right)

#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('s'):  # Press 's' to start capturing
#             if captured_images < num_images:
#                 # Save images
#                 left_image_path = os.path.join(output_dir, f"left_{captured_images + 1:02d}.jpg")
#                 right_image_path = os.path.join(output_dir, f"right_{captured_images + 1:02d}.jpg")
#                 cv2.imwrite(left_image_path, frame_left)
#                 cv2.imwrite(right_image_path, frame_right)
#                 print(f"Captured stereo pair {captured_images + 1}/{num_images}")
#                 captured_images += 1

#                 if captured_images >= num_images:
#                     print("All images captured.")
#                     break
#             else:
#                 print("Capture limit reached.")
#                 break
#         elif key == ord('q'):  # Press 'q' to quit
#             print("Exiting without capturing.")
#             break

#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()


# def main():
#     # Define the output directory
#     output_directory = "calibration_images"
    
#     # Capture stereo images
#     capture_stereo_images(output_directory, num_images=20, delay=1)


# if __name__ == "__main__":
#     main()


import cv2
import os
import time


def capture_stereo_images(output_dir, num_images=20, capture_interval=2):
    """
    Automatically capture stereo images for calibration.

    :param output_dir: Directory to save captured images.
    :param num_images: Number of stereo pairs to capture.
    :param capture_interval: Interval (in seconds) between captures.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Created output_dir:", output_dir)

    # Open the stereo camera
    cap = cv2.VideoCapture(1)  # Adjust if necessary for your stereo camera index

    if not cap.isOpened():
        print("Error: Could not open stereo camera.")
        return

    print(f"Starting capture. {num_images} stereo pairs will be saved to '{output_dir}'.")
    print("Ensure the checkerboard is visible in both camera views.")

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

        # Save the images
        left_image_path = os.path.join(output_dir, f"left_{captured_images + 1:02d}.jpg")
        right_image_path = os.path.join(output_dir, f"right_{captured_images + 1:02d}.jpg")
        cv2.imwrite(left_image_path, frame_left)
        cv2.imwrite(right_image_path, frame_right)
        print(f"Captured stereo pair {captured_images + 1}/{num_images}")

        captured_images += 1

        # Wait for the specified interval before capturing the next pair
        time.sleep(capture_interval)

        # Exit on key press (e.g., 'q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture terminated early by user.")
            break

    print("Capture complete. All images saved.")
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Define the output directory for saved images
    output_directory = "calibration_images"

    # Number of images and interval settings
    num_images_to_capture = 20  # Adjust as needed
    capture_interval_seconds = 10  # Time between captures in seconds

    # Start the capture process
    capture_stereo_images(output_directory, num_images=num_images_to_capture, capture_interval=capture_interval_seconds)


if __name__ == "__main__":
    main()
