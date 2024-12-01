import time
import hid

def setPosition(device, servos, position=None, duration=1000, wait=False):
    data = bytearray([1, duration & 0xff, (duration & 0xff00) >> 8])

    if position == None:
        raise ValueError('Parameter \'position\' missing.')
    if isinstance(position, int):
        if position < 500 or position > 2500:
            raise ValueError('Parameter \'position\' must be between 0 and 2500.')
    data.extend([servos, position & 0xff, (position & 0xff00) >> 8])

    CMD_SERVO_MOVE = 0x03
    send(device, CMD_SERVO_MOVE, data)

    if wait:
        time.sleep(duration/1000)

def send(device, cmd, data = []):
    HEADER = 0x55
    report_data = [
        0, 
        HEADER,
        HEADER,
        len(data) + 2,
        cmd
    ]
    
    if len(data):
        report_data.extend(data)
    
    print('Actually sending: ', report_data)
    # hex_list = [hex(x) for x in report_data]
    # print('Actually sending: ', hex_list, '\n')
    device.write(report_data)

def main():
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("    Running arm movements ...      ")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

    device = hid.device()
    device.open(0x0483, 0x5750) # LeArm Vendor ID: 0x483, LeArm Product ID: 0x5750
    device.set_nonblocking(1)
    print('Serial number:', device.get_serial_number_string())

    # Set servos 3-6 to 500 with separate commands
    duration = 2000
    setPosition(device, 3, 500, duration, False)
    setPosition(device, 4, 500, duration, False)
    setPosition(device, 5, 500, duration, False)
    setPosition(device, 6, 500, duration, False)

    # Sleep 10 seconds
    print('\nSleeping ...\n')
    time.sleep(10)

    # Set servos 3-6 to 1500 with 1 command
    print('Setting all servos to 1500 ...')
    data = [0, 85, 85, 17, 3, 4, 208, 7, 3, 220, 5, 4, 220, 5, 5, 220, 5, 6, 220, 5]
    device.write(data)

    device.close()

if __name__ == "__main__":
    main()

