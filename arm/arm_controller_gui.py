import tkinter as tk
from arm import Arm
import math


class ArmCanvasControlApp:
    def __init__(self, root, arm):
        self.root = root
        self.arm = arm

        # Set up GUI
        self.root.title("3D Arm Control - Semi-Circle GUI")
        self.root.geometry("500x500")

        # Canvas settings
        self.canvas_size = 400  # Size of the canvas (square)
        self.radius = 193       # Radius of the semi-circle (mm)
        self.scale = self.canvas_size / (2 * self.radius)  # Scale factor (pixels per mm)
        self.center = self.canvas_size // 2

        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Draw semi-circle boundary
        self.canvas.create_arc(
            10, 10, self.canvas_size - 10, self.canvas_size - 10,
            start=0, extent=180, outline="black", style=tk.ARC, width=2
        )

        # Create movable point
        self.point_radius = 5  # Radius of the draggable point
        self.point = self.canvas.create_oval(
            self.center - self.point_radius, self.center - self.point_radius,
            self.center + self.point_radius, self.center + self.point_radius,
            fill="red"
        )

        # Bind dragging events
        self.canvas.tag_bind(self.point, "<B1-Motion>", self.drag_point)

        # Reset button
        reset_button = tk.Button(self.root, text="Reset", command=self.reset_position)
        reset_button.grid(row=1, column=0, pady=10)

    def drag_point(self, event):
        """Handle dragging of the point."""
        # Convert canvas coordinates to relative X, Y
        dx = (event.x - self.center) / self.scale
        dy = -(event.y - self.center) / self.scale  # Invert Y-axis for correct orientation

        # Calculate distance and constrain to semi-circle
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.radius:
            scale = self.radius / distance
            dx *= scale
            dy *= scale

        if dy < 0:  # Constrain to semi-circle (Y >= 0)
            dy = 0

        # Update the canvas point position
        x_canvas = self.center + dx * self.scale
        y_canvas = self.center - dy * self.scale
        self.canvas.coords(
            self.point,
            x_canvas - self.point_radius, y_canvas - self.point_radius,
            x_canvas + self.point_radius, y_canvas + self.point_radius
        )

        # Update the arm's position
        self.arm.move_to_xy(dx, dy, execute=True)

    def reset_position(self):
        """Reset the point to the default position (center of the semi-circle)."""
        self.canvas.coords(
            self.point,
            self.center - self.point_radius, self.center - self.point_radius,
            self.center + self.point_radius, self.center + self.point_radius,
        )
        self.arm.move_to_xy(0, 0, execute=True)

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
    app = ArmCanvasControlApp(root, arm)

    try:
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
