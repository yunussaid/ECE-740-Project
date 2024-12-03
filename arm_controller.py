import tkinter as tk
from tkinter import ttk
from arm import Arm  # Assuming your Arm class is in arm.py
import threading

class ArmControlApp:
    def __init__(self, root, arm):
        self.root = root
        self.arm = arm

        # Set up GUI
        self.root.title("3D Arm Control")
        self.root.geometry("400x400")

        # StringVars to hold current value displays
        self.x_value = tk.StringVar()
        self.y_value = tk.StringVar()
        self.z_value = tk.StringVar()

        # Sliders for X, Y, and Z
        self.x_slider = self.create_slider("X Axis", -200, 200, 100, 0, self.x_value)
        self.y_slider = self.create_slider("Y Axis", -200, 200, 100, 1, self.y_value)
        self.z_slider = self.create_slider("Z Axis", -200, 200, 100, 2, self.z_value)

        # Button to reset arm
        reset_button = ttk.Button(self.root, text="Reset", command=self.reset_position)
        reset_button.grid(row=4, column=1, pady=10)

        # Start a separate thread for continuous arm updates
        self.running = True
        self.update_thread = threading.Thread(target=self.update_arm_position, daemon=True)
        self.update_thread.start()

    def create_slider(self, label, min_val, max_val, default, row, value_var):
        """Helper to create a slider with current value and bounds display."""
        # Axis Label
        ttk.Label(self.root, text=label).grid(row=row, column=0, padx=10, pady=10)

        # Slider
        slider = ttk.Scale(
            self.root, from_=min_val, to=max_val, orient="horizontal", length=200,
            command=lambda v: self.update_current_value(value_var, slider)
        )
        slider.set(default)
        slider.grid(row=row, column=1, padx=10, pady=10)

        # Current Value Label
        ttk.Label(self.root, textvariable=value_var).grid(row=row, column=2, padx=10, pady=10)
        value_var.set(f"{default} mm")  # Set initial value

        # Bounds Label
        ttk.Label(self.root, text=f"[{min_val}, {max_val}]").grid(row=row + 1, column=1, padx=10, pady=5)

        return slider

    def update_current_value(self, value_var, slider):
        """Update the current value display for a slider."""
        value_var.set(f"{int(slider.get())} mm")

    def reset_position(self):
        """Reset the arm to the default position."""
        self.x_slider.set(100)
        self.y_slider.set(100)
        self.z_slider.set(100)

    def update_arm_position(self):
        """Continuously update the arm position based on slider values."""
        while self.running:
            x = int(self.x_slider.get())
            y = int(self.y_slider.get())
            z = int(self.z_slider.get())

            try:
                self.arm.move_to_3D_coordinate(x, y, z)
            except ValueError as e:
                print(f"Out of range: {e}")

            # Update every 100 ms
            self.root.after(100)

    def on_close(self):
        """Handle the window closing."""
        self.running = False
        self.arm.disconnect()
        self.root.destroy()


def main():
    # Connect to the robotic arm
    arm = Arm()
    arm.connect()

    # Create and run the app
    root = tk.Tk()
    app = ArmControlApp(root, arm)

    try:
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
