"""
This example demonstrates core features of the graphics API without using pre-built scene templates.
The GLFW backend is used but the exact same processs is done with the DearPyGui backend just with DPG
specific input handling, window/context setup, and the DpgScene wrapper as shown in any of the DPG examples.

This shows how to:
- Create a custom scene from scratch
- Load multiple models
- Build a parent-child hierarchy
- Create transform nodes for organization
- Animate objects (orbiting camera)
- Use GlfwCameraMover for user control

Scene Structure:
- DataLogger model (with TriAxes attached)
- Embedded sensor model (with orbiting camera model)
  - Transform node (continuously rotates)
    - Camera model (offset from parent to allow orbiting the sensor)

Controls:
- WASD + Space/Shift: Move camera
- Mouse drag (right button): Rotate camera

Note: Models are made in units of meters, so the values used here are quite small to reflect real-world scale.
"""

import glfw

from yostlabs.graphics import GL_Context, resources
from yostlabs.graphics.core import Scene, Camera, ModelObject, TransformNode
from yostlabs.graphics.prefabs import TriAxesObject
from yostlabs.graphics.glfw import GlfwCameraMover

import yostlabs.math.quaternion as yl_quat
import time

# SETUP 3D VIEWER
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Initialize OpenGL context
GL_Context.init(window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, 
                window_title="Basic Usage Example", visible=True)

# Create the main scene
scene = Scene("Main Scene")
scene.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)

# Create and configure the camera
camera = Camera("Main Camera")
camera.set_viewport(WINDOW_WIDTH, WINDOW_HEIGHT)
camera.set_perspective(fov=90, aspect_ratio=WINDOW_WIDTH / WINDOW_HEIGHT, near=0.001, far=1000)
camera.set_position([0, 0, 0.080])  # Position camera back from origin
scene.set_camera(camera)

# Setup camera mover for user control
window = GL_Context.get_window()
camera_mover = GlfwCameraMover(camera, window, camera_speed=0.030, rotation_speed=1.0)

# Create DataLogger model (on the left side)
datalogger = ModelObject("DataLogger", resources.get_model_path('DL-3.obj'))
datalogger.set_position([-0.040, -0.020, 0])  # Position to the left
scene.add_child(datalogger)

# Add TriAxes to the DataLogger
axes = TriAxesObject("DataLogger Axes", stick_length=15.0, stick_width=0.5,
                     triangle_width=2.5, triangle_length=4.0)
axes.set_scale(0.003)
datalogger.add_child(axes)

# Create Embedded sensor model (on the right side)
embedded_sensor = ModelObject("Embedded Sensor", resources.get_model_path('EM-3.obj'))
embedded_sensor.set_position([0.040, -0.020, 0])  # Position to the right
scene.add_child(embedded_sensor)

#Scale the embedded to match the size of the datalogger for better visual comparison
max_dimension_dl = datalogger.model.get_global_max_dimension()
max_dimension_em = embedded_sensor.model.get_global_max_dimension()
embedded_sensor.set_scale(max_dimension_dl / max_dimension_em)

# Create a transform node that will continuously rotate (for orbiting effect)
orbit_transform = TransformNode("Orbit Transform")
embedded_sensor.add_child(orbit_transform)

# Create camera model that will orbit around the embedded sensor
camera_model = ModelObject("Camera Model", resources.get_model_path('Camera.obj'))
camera_model.set_scale(0.003)  # Scale the camera model
camera_model.set_position([0.025, 0, 0])  # Offset from parent (orbit radius)
camera_model.set_rotation_quat(yl_quat.quat_from_euler([90], 'y', degrees=True))  # Rotate to face the sensor
orbit_transform.add_child(camera_model)

# Rotation angle for the orbit animation
orbit_angle = 0.0
orbit_speed = 180  # Degrees per second

last_time = time.time()

# Main render loop
while not glfw.window_should_close(window):
    # Handle GLFW events
    glfw.poll_events()
    
    # Update camera position and rotation based on user input
    camera_mover.update_camera_pos()
    camera_mover.update_camera_rotation()
    
    #Time tracking for smooth animation regardless of frame rate
    cur_time = time.time()
    delta_time = cur_time - last_time
    last_time = cur_time

    # Animate the orbit - rotate the transform node around Z axis
    orbit_angle += orbit_speed * delta_time
    if orbit_angle >= 360.0:
        orbit_angle -= 360.0
    
    quat = yl_quat.quat_from_euler([orbit_angle], 'z', degrees=True)
    orbit_transform.set_rotation_quat(quat)
    
    # Render the scene
    scene.render()
    
    # Swap buffers
    glfw.swap_buffers(window)

GL_Context.cleanup()
