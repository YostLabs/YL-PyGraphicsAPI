"""
Example of visualizing a Threespace Sensor's orientation in a DearPyGui window.

Additional Requirements:
pip install yostlabs-graphics[dpg]
pip install yostlabs
or
pip install dearpygui
pip install yostlabs

Controls:
- WASD + Space/Shift: Move camera
- Mouse drag (right button): Rotate camera
"""

import dearpygui.dearpygui as dpg

from yostlabs.graphics import GL_Context, Font, resources
from yostlabs.graphics import ModelObject
from yostlabs.graphics import scene_prefabs
from yostlabs.graphics.dpg import DpgCameraMover, DpgScene

from yostlabs.math.axes import AxisOrder
from yostlabs.graphics import GL_AXIS_ORDER

#python -m pip install yostlabs
from yostlabs.tss3.api import ThreespaceSensor

#SETUP 3D VIEWER
TEXTURE_WIDTH = 600
TEXTURE_HEIGHT = 600

#NOTE: When using the DearPyGui window, the GL_Window is just a hidden context for rendering to a texture
#so its initial settings are not relevant. The visible window is created and managed by DearPyGui.
GL_Context.init()
#GL_Context.default_font = Font(resources.get_font_path('FiraCode-Regular')) #Optionally load a specified font

# Create model
model = ModelObject("DL", resources.get_model_path('DL-3.obj'))

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

#--------------Setup DearPyGui----------------
dpg.create_context()
dpg.create_viewport(width=TEXTURE_WIDTH+33, height=TEXTURE_HEIGHT+55) #With padding for window borders and title bar

with dpg.window() as primary_window:
    #dpg_scene.createDpgTexture() creates the internal DPG texture used by the image.
    #It must be called after dpg.create_context() and before adding the image to the window.
    #If displaying the texture in multiple places, only call createDpgTexture() once and use scene.get_dpg_texture()
    #to obtain the same texture object for use elsewhere. Repeated calls to createDpgTexture() will create a memory leak
    #if not manually managed.
    dpg.add_image(dpg_scene.createDpgTexture(), width=dpg_scene.texture_width, height=dpg_scene.texture_height)

dpg.set_primary_window(primary_window, True)

#--------------Setup Sensor--------------
#Auto Detect USB connection
sensor = ThreespaceSensor()
SENSOR_AXIS_ORDER = AxisOrder(sensor.get_settings("axis_order"))
orientation_scene.set_axis_order(SENSOR_AXIS_ORDER)

# Main render loop
dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    # Update camera using the camera mover
    orientation_scene.update_camera_pos()
    orientation_scene.update_camera_rotation()

    orientation = sensor.getTaredOrientation().data

    # Map sensor axes to OpenGL axes
    glQuat = SENSOR_AXIS_ORDER.swap_to(GL_AXIS_ORDER, orientation, rotational=True)
    model.set_rotation_quat(glQuat)
    
    # Render the scene and update the internal DPG texture
    dpg_scene.render()
    dpg_scene.update_dpg_texture()  
    
    # Swap buffers
    dpg.render_dearpygui_frame()

dpg.destroy_context()
GL_Context.cleanup()
