import tkinter as tk
from tkinter import ttk
from arm import Arm


class ArmControlApp:
    def __init__(self, root, arm):
        self.root = root
        self.arm = arm

        # Set up GUI
        self.root.title("3D Arm Control - Circular Bounds with Auto-Z")
        self.root.geometry("400x300")

        # StringVars to display current slider values
        self.x_value = tk.StringVar()
        self.y_value = tk.StringVar()

        # Sliders for X and Y
        self.x_slider = self.create_slider("X Axis", -193, 193, 0, 0, self.x_value)
        self.y_slider = self.create_slider("Y Axis", -193, 193, 193, 1, self.y_value)

        # Button to reset arm
        reset_button = ttk.Button(self.root, text="Reset", command=self.reset_position)
        reset_button.grid(row=2, column=1, pady=10)

        # Start a callback for live updates
        self.update_arm_position()

    def create_slider(self, label, min_val, max_val, default, row, value_var):
        """Helper to create a slider with a value display."""
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
        value_var.set(f"{default:.1f} mm")  # Set initial value

        return slider

    def update_current_value(self, value_var, slider):
        """Update the current value display for a slider."""
        value_var.set(f"{slider.get():.1f} mm")

    def reset_position(self):
        """Reset the sliders to default values."""
        self.x_slider.set(0)
        self.y_slider.set(0)

    def update_arm_position(self):
        """Continuously update the arm's position based on sliders."""
        x = self.x_slider.get()
        y = self.y_slider.get()

        # Move the arm to the specified (x, y) position
        try:
            self.arm.move_to_xy(x, y)
        except ValueError as e:
            print(f"Out of range: {e}")

        # Schedule the next update
        self.root.after(100, self.update_arm_position)

    def on_close(self):
        """Handle window close event."""
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
