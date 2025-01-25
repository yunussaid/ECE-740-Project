import cv2
import math
import os
import time

# Final results: It turns out that cv2.CAP_MSMF is slow during initialization but it has minimal lag during runtime.
# On the other hand, cv2.CAP_DSHOW is very quick to initalize but it's laggy during runtime. cv2.CAP_FFMPEG wasn't
# even able to access the camera at all.

def test_resolution_fps_combinations(video_capture_api, api_name, save_images=False):
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

    print(f"Testing Resolution, FPS, and Aspect Ratio combinations w/ {api_name}...")
    for width, height in resolutions:
        # Open the camera
        cap = cv2.VideoCapture(1, video_capture_api)

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
        # print()
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
    video_capture_apis = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_FFMPEG]
    api_name = ["CAP_MSMF", "CAP_DSHOW", "CAP_FFMPEG"]
    
    for api, name in zip(video_capture_apis, api_name):
        print(f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        start_time = time.time()
        test_resolution_fps_combinations(api, name, True)
        elapsed_time = time.time() - start_time
        print(f"Execution time w/ {name}: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()
