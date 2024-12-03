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
        self.DEFAULT_DURATION = 1000  # Default duration for servo movements (ms)

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
        self.set_all_angles([0, 0, 0, 0], duration)

    def move_to_3D_point_4_dof(self, x, y, z=0):
        """Move the arm to a specified 3D coordinate."""
        
        print(f"3D destination = (x:{x}, y:{y}, z:{z})")

        """X-Y Plane Calculations"""
        
        if y < 0:
            raise ValueError("Destiation is out of reach.")
        theta_base = math.degrees( math.atan2(x, y) )
        
        L = math.sqrt(x**2 + y**2)  # X-Y plane distance to dest
        print(f"L = {L:.2f}")

        """L-Z Plane Calculations"""

        if z < 0:
            raise ValueError("Destiation is out of reach.")
        h = math.sqrt(L**2 + z**2)
        print(f"h = {h:.2f}")
        
        L1 = 104    # Shoulder to elbow (in mm)
        L2 = 88.5   # Elbow to net servo (in mm)

        phi = math.degrees( math.atan2(z, L) )
        cos_theta = (h**2 + L1**2 - L2**2) / (2*h*L1)
        cos_alpha = (h**2 + L2**2 - L1**2) / (2*h*L2)
        theta = math.degrees( math.acos(cos_theta) )
        alpha = math.degrees( math.acos(cos_alpha) )
        print(f"phi = {phi:.2f}")
        print(f"theta = {theta:.2f}")
        print(f"alpha = {alpha:.2f}")

        theta_shoulder = 90 - phi - theta
        theta_elbow = theta + alpha

        """Final Angles"""

        theta_net = 0
        theta_net = (theta_shoulder + theta_elbow)
        if theta_net > 90: theta_net = 90
        print(f"Base:{theta_base:.2f}°, Shoulder:{theta_shoulder:.2f}°, Elbow:{theta_elbow:.2f}°, Net:{theta_net:.2f}°")
        
        set_servo_angles = True
        if set_servo_angles:
            self.set_angle(6, theta_base)  # Base
            self.set_angle(5, theta_shoulder)  # Shoulder
            self.set_angle(4, theta_elbow)  # Elbow
            self.set_angle(3, theta_net)  # Net parallelity

    def move_to_xy(self, x, y, debug=False):
        """
        Move the arm to a specified (x, y) coordinate in the X-Y plane,
        calculating the corresponding Z value automatically based on the sphere geometry.
        """
        # Arm constants
        radius = 193  # Sphere radius (maximum reach of the arm)

        # Ensure the point lies within the circle of radius 193 mm
        distance = math.sqrt(x**2 + y**2)
        if distance > radius:
            raise ValueError("Destination is out of reach. Ensure x^2 + y^2 <= 193^2.")

        # Calculate Z value using the sphere equation
        z = math.sqrt(radius**2 - distance**2)

        # Debug information
        if debug:
            print(f"Target position: (x: {x:.2f}, y: {y:.2f}, z: {z:.2f})")

        # Calculate servo angles
        # Base rotation (Servo 6)
        theta_base = math.degrees(math.atan2(x, y))

        # Shoulder tilt (Servo 5)
        theta_shoulder = math.degrees(math.asin(z / radius))

        # Net tilt (Servo 3) to maintain parallelity
        theta_net = theta_shoulder

        # Set the servo angles
        self.set_angle(6, theta_base)      # Base
        self.set_angle(5, theta_shoulder)  # Shoulder
        self.set_angle(4, 0)               # Fix Elbow at 0°
        self.set_angle(3, theta_net)       # Net tilt

        if debug:
            print(f"Servo angles: Base: {theta_base:.2f}°, Shoulder: {theta_shoulder:.2f}°, Net: {theta_net:.2f}°")


def main():
    arm = Arm()

    try:
        arm.connect()

        # Move to a specific 3D coordinate
        arm.move_to_xy(100, 100)

        # Sleep for 5 seconds
        print("\nSleeping for 5 seconds...")
        time.sleep(5)

        # Reset servos 3-6 to 0 degrees
        print("\nResetting servos to defualt angles...")
        arm.reset_all_angles()

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
