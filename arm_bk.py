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
        self.DEFAULT_DURATION = 1500  # Default duration for servo movements (ms)

        # Individual angle limits for each servo
        self.servo_angle_limits = {
            3: (-90, 90),   # Servo 3 (Net up/down)
            4: (-90, 90),   # Servo 4 (Elbow tilt)
            5: (-90, 90),   # Servo 5 (Shoulder tilt)
            6: (-90, 90)    # Servo 6 (Base rotation)
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

    def send(self, cmd, data=[]):
        """Send a command to the robotic arm."""
        HEADER = 0x55
        report_data = [
            0, HEADER, HEADER, len(data) + 2, cmd
        ]
        if len(data):
            report_data.extend(data)
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
        min_angle, max_angle = self.servo_angle_limits[servo]
        if angle < min_angle or angle > max_angle:
            raise ValueError(f"Servo {servo} angle must be between {min_angle} and {max_angle} degrees.")

        # Convert the angle to PWM and send the command
        pwm_position = self.angle_to_pwm(angle)
        data = bytearray([1, duration & 0xFF, (duration & 0xFF00) >> 8, servo, pwm_position & 0xFF, (pwm_position & 0xFF00) >> 8])
        CMD_SERVO_MOVE = 0x03
        self.send(CMD_SERVO_MOVE, data)

        if wait:
            time.sleep(duration / 1000)

    def set_all_angles(self, angles, duration=None, wait=False):
        """Set angles for multiple servos at once, respecting individual limits."""
        if duration is None:
            duration = self.DEFAULT_DURATION
        if len(angles) != 4:
            raise ValueError("Expected 4 angles for servos 3, 4, 5, and 6.")
        
        self.set_angle(3, angles[0], duration)
        self.set_angle(4, angles[1], duration)
        self.set_angle(5, angles[2], duration)
        self.set_angle(6, angles[3], duration)

        if wait:
            time.sleep(duration / 1000)
    
    def reset_all_angles(self, duration=None):
        """Reset angle to 0 for all servos."""
        self.set_all_angles([0, 0, 0, 0], duration)

    def moveToPos(self, x, y, z):
        print(f"moveToPos: x={x}, y={y}, z={z}")

        b = math.atan2(y, x) * (180 / math.pi)
        print(f"Base Angle: {b}")
        
        l = math.sqrt(x**2 + y**2)
        print(f"l: {l}")
        
        h = math.sqrt(l**2 + z**2)
        print(f"h: {h}")
        
        phi = math.atan2(z, l) * (180 / math.pi)
        theta = math.acos((105**2 + h**2 - 90**2) / (2 * 105 * h)) * (180 / math.pi)
        print(f"Phi: {phi}")
        print(f"Theta: {theta}")

        a1 = phi + theta
        a2 = phi - theta
        print(f"a1: {a1}")
        print(f"a2: {a2}")

        s4 = -1*(a1 - a2)
        s5 = 90 - a1

        print(f"s4: {s4}")
        print(f"s5: {s5}")

        self.set_all_angles([0, s4, s5, b], 200, True)

def main():
    arm = Arm()

    try:
        arm.connect()

        # Set servos 3-6 to 0 degrees
        print("\nSetting servos to custom angles...")
        # arm.moveToPos(100, 100, 170)
        # arm.moveToPos(100, 100, 160)
        # arm.moveToPos(100, 100, 150)
        # arm.moveToPos(100, 100, 140)
        # arm.moveToPos(100, 100, 130)
        arm.moveToPos(100, 100, 120)
        arm.moveToPos(100, 100, 110)
        arm.moveToPos(100, 100, 100)
        arm.moveToPos(100, 100, 90)
        arm.moveToPos(100, 100, 80)
        arm.moveToPos(100, 100, 70)
        arm.moveToPos(100, 100, 60)
        arm.moveToPos(100, 100, 50)
        arm.moveToPos(100, 100, 40)
        arm.moveToPos(100, 100, 30)
        arm.moveToPos(100, 100, 20)
        arm.moveToPos(100, 100, 10)
        arm.moveToPos(100, 100, 0)


        # Sleep for 5 seconds
        print("\nSleeping for 5 seconds...")
        time.sleep(5)

        # # Set servos 3-6 to custom angles within limits
        # print("\nResetting all servos...")
        # arm.reset_all_angles()

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
