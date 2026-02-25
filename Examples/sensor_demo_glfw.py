"""
Example of visualizing a Threespace Sensor's orientation in a GLFW window.

Additional Requirements:
pip install yostlabs

Controls:
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

#python -m pip install yostlabs
from yostlabs.tss3.api import ThreespaceSensor

#SETUP 3D VIEWER
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

#IMPORTANT: When using the GLFW window, set visible to True
GL_Context.init(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, window_title="3D Viewer", visible=True)
#GL_Context.default_font = Font(resources.get_font_path('FiraCode-Regular')) #Optionally load a specified font

# Create model
model = ModelObject("DL", resources.get_model_path('DL-3.obj'))

# Create OrientationScene with model and axes
orientation_scene = scene_prefabs.OrientationScene(
    WINDOW_WIDTH, 
    WINDOW_HEIGHT, 
    model, 
    font=GL_Context.default_font,
    name="Main Scene"
)
orientation_scene.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)

# Get the GLFW window and shader program
window = GL_Context.get_window()

# Set up GLFW camera mover for the orientation scene
glfw_camera_mover = GlfwCameraMover(orientation_scene.camera, window)
orientation_scene.set_camera_mover(glfw_camera_mover)

# SENSOR INFO
#Auto Detect USB connection
sensor = ThreespaceSensor()
SENSOR_AXIS_ORDER = AxisOrder(sensor.get_settings("axis_order"))
orientation_scene.set_axis_order(SENSOR_AXIS_ORDER)

# Main render loop
while not glfw.window_should_close(window):
    # Handle GLFW events
    glfw.poll_events()
    
    # Update camera using the camera mover
    orientation_scene.update_camera_pos()
    orientation_scene.update_camera_rotation()

    orientation = sensor.getTaredOrientation().data

    # Map sensor axes to OpenGL axes
    glQuat = SENSOR_AXIS_ORDER.swap_to(GL_AXIS_ORDER, orientation, rotational=True)
    model.set_rotation_quat(glQuat)
    
    # Render the scene directly to the window
    orientation_scene.render()
    
    # Swap buffers
    glfw.swap_buffers(window)

GL_Context.cleanup()
