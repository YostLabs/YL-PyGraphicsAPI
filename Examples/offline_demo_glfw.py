"""
Example of controlling a 3D model's orientation manually in a GLFW window.
This demo allows keyboard-based rotation control without requiring a sensor.

Controls:
- Arrow Keys (Left/Right): Rotate around the first axis in the euler_str order (Default: Z)
- Arrow Keys (Up/Down): Rotate around the second axis in the euler_str order (Default: Y)
- Q/E: Rotate around the third axis in the euler_str order (Default: X)
- WASD + Space/Shift: Move camera
- Mouse drag (right button): Rotate camera
"""

import glfw

from yostlabs.graphics import GL_Context, Font, resources
from yostlabs.graphics import ModelObject
from yostlabs.graphics import scene_prefabs
from yostlabs.graphics.glfw import GlfwCameraMover

from yostlabs.math.axes import AxisOrder
from yostlabs.graphics import GL_AXIS_ORDER
from yostlabs.math.quaternion import quat_from_euler

#SETUP 3D VIEWER
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

#IMPORTANT: When using the GLFW window, set visible to True
GL_Context.init(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, window_title="3D Viewer - Offline Demo", visible=True)
#GL_Context.default_font = Font(resources.get_font_path('FiraCode-Regular')) #Optionally load a specified font

# Create model
model = ModelObject("EM", resources.get_model_path('EM-3.obj'))

# Create OrientationScene with model and axes
orientation_scene = scene_prefabs.OrientationScene(
    WINDOW_WIDTH, 
    WINDOW_HEIGHT, 
    model, 
    font=GL_Context.default_font,
    name="Main Scene"
)
orientation_scene.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)

# SENSOR INFO - Using manual control instead of real sensor
SENSOR_AXIS_ORDER = AxisOrder("NED") #Example non default axis order, North East Down.
euler_str = "zyx"
orientation_scene.set_axis_order(SENSOR_AXIS_ORDER)

# Variables for manual Euler angle control (modified via keyboard)
euler_angles = [0.0, 0.0, 0.0]  # Angles in degrees for euler_str order

# Get the GLFW window
window = GL_Context.get_window()

# Set up GLFW camera mover for the orientation scene
glfw_camera_mover = GlfwCameraMover(orientation_scene.camera, window)
orientation_scene.set_camera_mover(glfw_camera_mover)

# Main render loop
while not glfw.window_should_close(window):
    # Handle GLFW events
    glfw.poll_events()
    
    # Update camera using the camera mover
    orientation_scene.update_camera_pos()
    orientation_scene.update_camera_rotation()
    
    # Handle keyboard input for rotating the model
    # Arrow keys adjust euler angles
    if glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS:
        euler_angles[0] -= 1.0
    if glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS:
        euler_angles[0] += 1.0
    if glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS:
        euler_angles[1] += 1.0
    if glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS:
        euler_angles[1] -= 1.0
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        euler_angles[2] -= 1.0 
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        euler_angles[2] += 1.0
    
    # Convert Euler angles to quaternion
    rotation = quat_from_euler(euler_angles, euler_str, degrees=True)

    # Map sensor axes to OpenGL axes
    glQuat = SENSOR_AXIS_ORDER.swap_to(GL_AXIS_ORDER, rotation, rotational=True)
    model.set_rotation_quat(glQuat)
    
    # Render the scene directly to the window
    orientation_scene.render()
    
    # Swap buffers
    glfw.swap_buffers(window)

GL_Context.cleanup()
