import tkinter as tk
from arm import Arm
import math


class NetCanvasControlApp:
    def __init__(self, root, arm):
        self.root = root
        self.arm = arm

        # Set up GUI
        self.root.title("Net Canvas Control - Semi-Circle GUI")
        self.root.geometry("600x650")

        # Canvas settings
        self.canvas_size = 500  # Size of the canvas (square)
        self.outer_radius = 320  # Radius of the outer semi-circle (193 arm reach + 127 net length)
        self.inner_radius = 127  # Radius of the inner semi-circle (net length)
        self.scale = self.canvas_size / (2 * self.outer_radius)  # Scale factor (pixels per mm)
        self.center = self.canvas_size // 2

        # Create canvas
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        # Draw outer semi-circle boundary
        self.canvas.create_arc(
            10, 10, self.canvas_size - 10, self.canvas_size - 10,
            start=0, extent=180, outline="black", style=tk.ARC, width=2
        )

        # Draw inner semi-circle boundary
        inner_radius_scaled = self.inner_radius * self.scale
        self.canvas.create_arc(
            self.center - inner_radius_scaled, self.center - inner_radius_scaled,
            self.center + inner_radius_scaled, self.center + inner_radius_scaled,
            start=0, extent=180, outline="gray", style=tk.ARC, width=1, dash=(4, 4)
        )

        # Create movable point
        self.point_radius = 5  # Radius of the draggable point
        self.point = self.canvas.create_oval(
            self.center - self.point_radius, self.center - self.point_radius - int(self.inner_radius * self.scale),
            self.center + self.point_radius, self.center + self.point_radius - int(self.inner_radius * self.scale),
            fill="red"
        )

        # Live coordinate display
        self.coordinates_label = tk.Label(self.root, text="Net Coordinates: (x: 0.0 mm, y: 127.0 mm)")
        self.coordinates_label.grid(row=1, column=0, pady=10)

        # Bind dragging events
        self.canvas.tag_bind(self.point, "<B1-Motion>", self.drag_point)

        # Reset button
        reset_button = tk.Button(self.root, text="Reset", command=self.reset_position)
        reset_button.grid(row=2, column=0, pady=10)

    def drag_point(self, event):
        """Handle dragging of the point."""
        # Convert canvas coordinates to relative X, Y
        dx = (event.x - self.center) / self.scale
        dy = -(event.y - self.center) / self.scale  # Invert Y-axis for correct orientation

        # Calculate distance and constrain to outer semi-circle
        distance = math.sqrt(dx**2 + dy**2)
        if distance > self.outer_radius:
            scale = self.outer_radius / distance
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

        # Update the live coordinate display
        self.coordinates_label.config(text=f"Net Coordinates: (x: {dx:.1f} mm, y: {dy:.1f} mm)")

        # Update the arm's position
        try:
            self.arm.move_net_to_xy(dx, dy, execute=True, debug=True)
        except ValueError as e:
            print(f"Out of range: {e}")

    def reset_position(self):
        """Reset the point to the default position (center of the semi-circle)."""
        self.canvas.coords(
            self.point,
            self.center - self.point_radius, self.center - self.point_radius - int(self.inner_radius * self.scale),
            self.center + self.point_radius, self.center + self.point_radius - int(self.inner_radius * self.scale),
        )
        self.coordinates_label.config(text="Net Coordinates: (x: 0.0 mm, y: 127.0 mm)")
        self.arm.move_net_to_xy(0, self.inner_radius, execute=True, debug=True)

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
    app = NetCanvasControlApp(root, arm)

    try:
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
