from camera.camera import Camera
from arm.arm import Arm
import time
import cv2

def main():
    camera = Camera(show_frames=True, show_masks=True, dynamic_mode=True)
    arm = Arm()
    arm.connect()
    print("Camera and Arm connected.")

    while True:
        ball_position = camera.get_ball_position()
        if ball_position:
            print(f"Ball Position:  [{''.join(f'{coord:>6}' for coord in ball_position)}]")
            
            x = ball_position[0]
            y = ball_position[2] - 390
            if x >= 200 or x <= -200:
                y -= 100 # offset beacuse Y gets big as X gets big
            print(f"Move Net to XY: [{x}, {y}]\n")
            arm.move_net_to_xy(x_net=x, y_net=y, duration=150, execute=True)

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
