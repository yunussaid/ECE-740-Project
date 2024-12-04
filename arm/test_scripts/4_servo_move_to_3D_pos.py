import math

# x axis is right-left axis (right is positive x)
# y axis is front-back axis (front is positive y)
# z axis is up-down axis (up is positive z)

def moveToPosOG(x, y, z):
    print(f"moveToPos: x={x}, y={y}, z={z}")

    # Calculate the base angle
    b = math.atan2(y, x) * (180 / math.pi)
    print(f"Base Angle: {b}")
    
    # Calculate the x and y extension
    l = math.sqrt(x**2 + y**2)
    print(f"l: {l}")
    
    # Calculate the hypotenuse in 3D
    h = math.sqrt(l**2 + z**2)
    print(f"h: {h}")
    
    # Calculate phi and theta angles
    phi = math.atan2(z, l) * (180 / math.pi)
    theta = math.acos((h/2) / 75) * (180 / math.pi)  # Assuming the arm length is 75 units
    print(f"Phi: {phi}")
    print(f"Theta: {theta}")

    # Calculate angles for the arm's first and second parts
    a1 = phi + theta
    a2 = phi - theta

    # Call moveToAngle with calculated angles
    moveToAngle(b, a1, a2)

def moveToAngle(b, a1, a2):
    # This function would control the arm's motors or servos
    # print(f"Base Angle: {b}")
    print(f"Angle 1: {a1}")
    print(f"Angle 2: {a2}\n")

# Example usage
# moveToPosOG(53.033, 53.033, 66.144)


def moveToPos(x, y, z):
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

    moveToAngle(b, a1, a2)

moveToPos(4, 4, 40)