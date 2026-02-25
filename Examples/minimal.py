
import glfw

from yostlabs.graphics import GL_Context
from yostlabs.graphics import ModelObject, Font, resources
from yostlabs.graphics.scene_prefabs import OrientationScene

GL_Context.init(window_width=600, 
                window_height=600, 
                window_title="3D Viewer", 
                visible=True)  # Initialize the OpenGL context

# Load resources from the package
model_path = resources.get_model_path('DL-3.obj')  # Auto-finds Camera.obj in models/Camera/
font_path = resources.get_font_path('arial.ttf')

# Create objects
model = ModelObject("MyModel", model_path)
font = Font(font_path)

# Create a 3D scene with orientation visualization
scene = OrientationScene(600, 600, model, font=font)

# Get the GLFW window
window = GL_Context.get_window()
while not glfw.window_should_close(window):
    glfw.poll_events()
    scene.render()
    glfw.swap_buffers(window)

GL_Context.cleanup()