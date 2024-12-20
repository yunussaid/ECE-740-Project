import cv2
import math
import os

def test_resolution_fps_combinations(save_images=False):
    # Resolutions options from manufacturer (width, height)
    resolutions = [
        (640, 240),     # Expected FPS: 120fps
        (1280, 480),    # Expected FPS: 120fps
        (1600, 600),    # Expected FPS: 120fps
        (2560, 720),    # Expected FPS: 60fps
        (3200, 1200)    # Expected FPS: 60fps
    ]

    # Set max FPS
    max_fps = 120

    # Define the output directory for saved images
    output_dir = "resolution_vs_fps"

    print("Testing Resolution, FPS, and Aspect Ratio combinations ...")
    for width, height in resolutions:
        
        # Open the camera
        cap = cv2.VideoCapture(1)
        # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera. Check camera connection and index.")
            cap.release()
            return

        # Set the resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Attempt to set the target FPS
        cap.set(cv2.CAP_PROP_FPS, max_fps)

        # Get the actual resolution set by the camera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Get the actual FPS set by the camera
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate and format the aspect ratio in reduced form
        if actual_height != 0:
            gcd = math.gcd(actual_width, actual_height)
            aspect_ratio = f"{actual_width // gcd}:{actual_height // gcd}"
        else:
            aspect_ratio = "Error: Undefined Aspect Ratio (Height is 0)"

        # Print the results
        print()
        print(f"Requested: {width}x{height}p\tFPS: {max_fps:.0f}")
        print(f"Actual:    {actual_width}x{actual_height}p\tFPS: {actual_fps:.0f} \tAspect Ratio: {aspect_ratio}")

        # Save a sample image if flag is enabled
        if save_images:
            # Capture a frame
            ret, frame = cap.read()
            if ret:
                # Set a descriptive file name
                filename = f"{actual_width}x{actual_height}p_{int(actual_fps)}fps.jpg"

                # Create output directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                # Save the image to output directory
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                print(f"Saved image named {filename}")
            else:
                print("Error: Failed to capture frame. Skipping...")

        cap.release()

    if save_images:
        print(f"\nTesting complete. Test images saved at ./{output_dir}")


def main():
    test_resolution_fps_combinations(True)


if __name__ == "__main__":
    main()
