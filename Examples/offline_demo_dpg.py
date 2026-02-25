"""
Example of controlling a 3D model's orientation manually in a DearPyGui window.
This demo allows slider-based rotation control without requiring a sensor.

Additional Requirements:
pip install yostlabs-graphics[dpg]
or
pip install dearpygui

Controls:
- Euler angle sliders: Rotate model
- WASD + Space/Shift: Move camera
- Mouse drag (right button): Rotate camera
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import dearpygui.dearpygui as dpg

from yostlabs.graphics import GL_Context, Font, resources
from yostlabs.graphics import ModelObject
from yostlabs.graphics import scene_prefabs
from yostlabs.graphics.dpg import DpgCameraMover, DpgScene

from yostlabs.math.axes import AxisOrder
from yostlabs.graphics import GL_AXIS_ORDER
from yostlabs.math.quaternion import quat_from_euler

#SETUP 3D VIEWER
TEXTURE_WIDTH = 600
TEXTURE_HEIGHT = 600

#NOTE: When using the DearPyGui window, the GL_Window is just a hidden context for rendering to a texture
#so its initial settings are not relevant. The visible window is created and managed by DearPyGui.
GL_Context.init()
#GL_Context.default_font = Font(resources.get_font_path('FiraCode-Regular')) #Optionally load a specified font

# Create model
model = ModelObject("EM", resources.get_model_path('EM-3.obj'))

# Create OrientationScene with model and axes
orientation_scene = scene_prefabs.OrientationScene(
    TEXTURE_WIDTH, 
    TEXTURE_HEIGHT, 
    model, 
    font=GL_Context.default_font,
    name="Main Scene"
)
orientation_scene.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)

# Set up DPG camera mover for the orientation scene
dpg_camera_mover = DpgCameraMover(orientation_scene.camera)
orientation_scene.set_camera_mover(dpg_camera_mover)

# DPG requires an additional scene wrapper to allow rendering to a DPG Texture.
dpg_scene = DpgScene(TEXTURE_WIDTH, TEXTURE_HEIGHT, scene=orientation_scene)

# SENSOR INFO - Using manual control instead of real sensor
SENSOR_AXIS_ORDER = AxisOrder("NED") #Example non default axis order, North East Down.
euler_str = "zyx"
orientation_scene.set_axis_order(SENSOR_AXIS_ORDER)

#--------------Setup DearPyGui----------------
dpg.create_context()
dpg.create_viewport(width=TEXTURE_WIDTH+33, height=TEXTURE_HEIGHT+55+23) #With padding for window borders, title bar, and slider height

with dpg.window() as primary_window:
    euler_sliders = dpg.add_slider_floatx(size=3, clamped=True, min_value=-180, max_value=180, label=f"Angles {euler_str.upper()}")
    #dpg_scene.createDpgTexture() creates the internal DPG texture used by the image.
    #It must be called after dpg.create_context() and before adding the image to the window.
    #If displaying the texture in multiple places, only call createDpgTexture() once and use scene.get_dpg_texture()
    #to obtain the same texture object for use elsewhere. Repeated calls to createDpgTexture() will create a memory leak
    #if not manually managed.
    dpg.add_image(dpg_scene.createDpgTexture(), width=dpg_scene.texture_width, height=dpg_scene.texture_height)

dpg.set_primary_window(primary_window, True)

# Main render loop
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    # Update camera using the camera mover
    orientation_scene.update_camera_pos()
    orientation_scene.update_camera_rotation()

    # Rotate the model based on the Euler sliders
    euler = dpg.get_value(euler_sliders)[:3]
    rotation = quat_from_euler(euler, euler_str, degrees=True)

    # Map sensor axes to OpenGL axes
    glQuat = SENSOR_AXIS_ORDER.swap_to(GL_AXIS_ORDER, rotation, rotational=True)
    model.set_rotation_quat(glQuat)
    
    # Render the scene and update the internal DPG texture
    dpg_scene.render()
    dpg_scene.update_dpg_texture()  
    
    # Swap buffers
    dpg.render_dearpygui_frame()

dpg.destroy_context()
GL_Context.cleanup()
