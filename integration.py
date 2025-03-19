from collections import deque
from camera.camera import Camera
from arm.arm import Arm
import time
import cv2

def main():
    camera = Camera(show_frames=True, show_masks=True, dynamic_mode=True)
    arm = Arm()
    arm.connect()
    print("Camera and Arm connected.")

    # Initialize a deque to store the last 5 coordinates for the moving average
    coord_history = deque(maxlen=5)

    while True:
        ball_position = camera.get_ball_position()
        if ball_position:
            print(f"Ball Position:  [{''.join(f'{coord:>6}' for coord in ball_position)}]")
            
            x = ball_position[0]
            y = ball_position[2] - 390
            if x >= 200 or x <= -200:
                y -= 100  # offset because Y gets big as X gets big

            # Append the new coordinates to the deque
            coord_history.append((x, y))

            # Calculate the moving average of the last 5 coordinates
            avg_x = int(sum(coord[0] for coord in coord_history) / len(coord_history))
            avg_y = int(sum(coord[1] for coord in coord_history) / len(coord_history))

            print(f"Move Net to XY (Averaged): [{avg_x}, {avg_y}]\n")
            arm.move_net_to_xy(x_net=avg_x, y_net=avg_y, duration=150, execute=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # time.sleep(0.1) # for slowed down testing
    
    # Release all resources
    arm.reset_all_angles()
    arm.disconnect()
    camera.release()
    print("All resources have been released.")

if __name__ == "__main__":
    main()
