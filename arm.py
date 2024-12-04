import time
import hid
import math

class Arm:
    def __init__(self):
        self.device = None
        self.VENDOR_ID = 0x0483  # LeArm Vendor ID
        self.PRODUCT_ID = 0x5750  # LeArm Product ID
        self.SERVO_PWM_MIN = 500
        self.SERVO_PWM_MAX = 2500
        self.ANGLE_MIN = -90
        self.ANGLE_MAX = 90
        self.DEFAULT_DURATION = 350  # Default duration for servo movements (ms)

        # Individual angle limits for each servo
        # Servo 6 (Base rotation): range is -90° (max left rotation) to 90° (max right rotation)
        # Servo 5 (Shoulder tilt): range is -90° (max back rotation) to 90° (max front rotation)
        # Servo 4 (Elbow tilt): range is -90° (max back rotation) to 90° (max front rotation)
        # Servo 3 (Net up/down tilt): range is -90° (max down rotation) to 90° (max up rotation)
        self.servo_angle_limits = {
            3: (-90, 90, 1),    # Servo 3 (Net up/down tilt)
            4: (-90, 90, -1),   # Servo 4 (Elbow tilt)
            5: (-90, 90, 1),    # Servo 5 (Shoulder tilt)
            6: (-90, 90, -1)    # Servo 6 (Base rotation)
        }

    def connect(self):
        """Connect to the robotic arm."""
        self.device = hid.device()
        self.device.open(self.VENDOR_ID, self.PRODUCT_ID)
        self.device.set_nonblocking(1)
        print('Connected to arm with serial number:', self.device.get_serial_number_string())

    def disconnect(self):
        """Disconnect the robotic arm."""
        if self.device:
            self.device.close()
            print("Disconnected from arm.")

    def send(self, cmd, data=[], debug=False):
        """Send a command to the robotic arm."""
        HEADER = 0x55
        report_data = [
            0, HEADER, HEADER, len(data) + 2, cmd
        ]
        if len(data):
            report_data.extend(data)
        if debug:
            print('Sending:', report_data)
        self.device.write(report_data)

    def angle_to_pwm(self, angle):
        """Convert an angle (-90 to 90 degrees) to a PWM value (500 to 2500)."""
        pwm = int((angle - self.ANGLE_MIN) / (self.ANGLE_MAX - self.ANGLE_MIN) * 
                  (self.SERVO_PWM_MAX - self.SERVO_PWM_MIN) + self.SERVO_PWM_MIN)
        return pwm

    def set_angle(self, servo, angle, duration=None, wait=False):
        """ Set servo angle, respecting individual limits. """
        if duration is None:
            duration = self.DEFAULT_DURATION

        if servo not in self.servo_angle_limits:
            raise ValueError(f"Servo {servo} cannot be identified.")

        # Enforce individual angle limits for the servo
        min_angle, max_angle, orientatioin = self.servo_angle_limits[servo]
        if angle < min_angle or angle > max_angle:
            raise ValueError(f"Servo {servo} angle must be between {min_angle} and {max_angle} degrees.")

        # Convert the angle to PWM and send the command
        pwm_position = self.angle_to_pwm(angle*orientatioin)
        data = bytearray([1, duration & 0xFF, (duration & 0xFF00) >> 8, servo, pwm_position & 0xFF, (pwm_position & 0xFF00) >> 8])
        CMD_SERVO_MOVE = 0x03
        self.send(CMD_SERVO_MOVE, data)

        if wait:
            time.sleep(duration / 1000)

    def set_all_angles(self, angles, duration=None):
        """Set angles for multiple servos at once, respecting individual limits."""
        if duration is None:
            duration = self.DEFAULT_DURATION
        if len(angles) != 4:
            raise ValueError("Expected 4 angles for servos 3, 4, 5, and 6.")
        
        self.set_angle(3, angles[0], duration)
        self.set_angle(4, angles[1], duration)
        self.set_angle(5, angles[2], duration)
        self.set_angle(6, angles[3], duration)
    
    def reset_all_angles(self, duration=None):
        """Reset angle to 0 for all servos."""
        print("\nResetting servos to defualt angles...")
        self.set_all_angles([0, 0, 0, 0], duration)

    def rest(self, seconds):
        print(f"\nSleeping for {seconds} seconds...")
        time.sleep(seconds)

    def move_to_xy(self, x, y, execute = False, debug=False):
        """
        Move the arm to a specified (x, y) coordinate in the X-Y plane,
        calculating the corresponding Z value automatically based on the sphere geometry.
        """
        if y < 0:
            y = 0

        dist_3d = 193
        dist_2d = math.sqrt(x**2 + y**2)

        # Constrain to an x-y plane semi-circle of radius 193
        while dist_2d > dist_3d:
            scale = dist_3d / dist_2d
            x *= scale
            y *= scale
            dist_2d = math.sqrt(x**2 + y**2)
        
        z = math.sqrt(dist_3d**2 - dist_2d**2)
        if debug:
            print(f"\nTarget position 3D: (x: {x:.1f}, y: {y:.1f}, z: {z:.1f})")
        
        theta_base = math.degrees(math.atan2(x, y))
        theta_shoulder = math.degrees(math.acos(z / dist_3d))
        theta_net = theta_shoulder

        if debug:
            print(f"Servo angles: Base: {theta_base:.1f}°, Shoulder: {theta_shoulder:.1f}°, Net: {theta_net:.1f}°")

        if execute:
            self.set_angle(6, theta_base)      # Base
            self.set_angle(5, theta_shoulder)  # Shoulder
            self.set_angle(4, 0)               # Fix Elbow at 0°
            self.set_angle(3, theta_net)       # Net tilt

    def move_net_to_xy(self, x_net, y_net, execute=False, debug=False):
        """
        Move the center of the net to the specified (x_net, y_net) coordinate.
        The method accounts for the net's offset from Servo 3 and translates
        the target position accordingly before calling `move_to_xy`.
        """
        # Length of the net offset
        net_length = 120  # mm

        # Calculate the corresponding (x_servo3, y_servo3) for Servo 3
        dist_2d_net = math.sqrt(x_net**2 + y_net**2)

        if dist_2d_net < net_length:
            if y_net < 0:
                y_net = 0
            
            theta_base = math.degrees(math.atan2(x_net, y_net))
            if execute:
                self.set_angle(6, theta_base)

            if debug:
                print(f"\nNet target position: (x_net: {x_net:.1f}, y_net: {y_net:.1f})")
                print(f"Inside inner boundary, just setting base angle: {theta_base:.1f}°")
            
            return

        # Scale back to the valid position for Servo 3
        scale = (dist_2d_net - net_length) / dist_2d_net
        x_servo3 = x_net * scale
        y_servo3 = y_net * scale

        if debug:
            print(f"\nNet target position: (x_net: {x_net:.1f}, y_net: {y_net:.1f})")
            print(f"Translated Servo 3 position: (x_servo3: {x_servo3:.1f}, y_servo3: {y_servo3:.1f})")

        # Call move_to_xy with the translated coordinates
        self.move_to_xy(x_servo3, y_servo3, execute, debug)


def main():
    arm = Arm()

    try:
        arm.connect()

        arm.move_to_xy(-193, 0, True, True)
        arm.rest(10)
        arm.move_to_xy(0, 193, True, True)
        arm.rest(10)
        arm.move_to_xy(193, 0, True, True)
        arm.rest(10)

        arm.move_to_xy(68, 68, True, True)
        arm.rest(10)
        arm.move_to_xy(-68, 68, True, True)
        arm.rest(10)

        arm.move_to_xy(136.4, 136.4, True, True)
        arm.rest(10)
        arm.move_to_xy(-136.4, 136.4, True, True)
        arm.rest(10)
        
        arm.reset_all_angles()

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()