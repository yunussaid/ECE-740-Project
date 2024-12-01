import xarm
import time

arm = xarm.Controller('USB', debug=True)

# print('Battery voltage in volts:', arm.getBatteryVoltage())

duration=1000

arm.setPosition(6, 1000, duration, wait=True)
time.sleep(2)

arm.setPosition(6, 900, duration, wait=True)
time.sleep(2)

arm.setPosition(6, 800, duration, wait=True)
time.sleep(2)

arm.setPosition(6, 700, duration, wait=True)
time.sleep(2)

arm.setPosition(6, 600, duration, wait=True)
time.sleep(2)

arm.setPosition(6, 500, duration, wait=True)
time.sleep(2)

# arm.servoOff()