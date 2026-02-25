# yostlabs-graphics

This package is part of the yostlabs python package system. It provides a simple 3D graphics engine utilizing OpenGL. It is intended for use with YostLabs orientation sensors to visualize data, but simply supports loading basic 3D models into a scene graph, allowing for transform manipulation, and rendering. It alo has support for integrating with [DearPyGui](https://github.com/hoffstadt/DearPyGui), a python library that allows quickly creating GUIs for rapid prototyping.

## Features

- **Scene Graph**: Hierarchical game object system with parent-child transforms
- **Core Objects**: Camera, Scene, ModelObject (OBJ files), TextMesh(BillBoard), ArrowObject, HudOverlay
- **Prefab Objects**: TriAxesObject, LabeledTriAxesObject for axis visualization
- **Pre-built Scenes**: CameraScene, OrientationScene with automatic camera setup
- **OpenGL Context**: GL_Context for window and OpenGL initialization
- **DearPyGui Integration**: Render to a DearPyGui window via the texture system

## Installation

```bash
pip install yostlabs-graphics
```

Optional dependencies:
```bash
pip install yostlabs-graphics[dpg]      # DearPyGui backend support
pip install yostlabs-graphics[sensor]   # Yost Labs sensor integration
```

## Usage

```python
import glfw
from yostlabs.graphics import GL_Context
from yostlabs.graphics import ModelObject, Font, resources
from yostlabs.graphics.scene_prefabs import OrientationScene

# Initialize the OpenGL context
GL_Context.init(window_width=600, 
                window_height=600, 
                window_title="3D Viewer", 
                visible=True)

# Load resources from the package (fonts and models are bundled)
model_path = resources.get_model_path('DL-3.obj')
font_path = resources.get_font_path('arial.ttf')

# Create objects
model = ModelObject("MyModel", model_path)
font = Font(font_path)

# Create a 3D scene with orientation visualization
scene = OrientationScene(600, 600, model, font=font)

# Main rendering loop
window = GL_Context.get_window()
while not glfw.window_should_close(window):
    glfw.poll_events()
    scene.render()
    glfw.swap_buffers(window)

GL_Context.cleanup()
```

For additional examples of constructing scenes, integrating with sensors, and DearPyGUI support,
please refer to the [Examples Folder](./Examples/).

### Resource Access

The package bundles fonts and 3D models that can be accessed via the resources module:

```python
from yostlabs.graphics import resources

# Get path to a system or bundled font
font_path = resources.get_font_path('arial.ttf')

# Get path to a bundled model
model_path = resources.get_model_path('DL-3.obj')

# List all available resources
fonts = resources.list_available_fonts()       # ['arial.ttf', ...]
models = resources.list_available_models()     # {'Camera': ['Camera.obj'], 'DataLogger': [...], ...}
```

**Note:** All bundled sensor models (DataLogger, Embedded) are modeled in units of meters.

## Sensor Animation Utility

The package includes a ready-to-use tool for visualizing Threespace Sensor orientation in real-time. This utility automatically connects to a Threespace Sensor and displays its 3D orientation with an appropriate sensor model.

**Requirements:**
```bash
pip install yostlabs-graphics[sensor]
```

**Command-line usage:**
```bash
# Auto-discover sensor and display tared orientation
python -m yostlabs.tools.animate_sensor

# Display untared orientation
python -m yostlabs.tools.animate_sensor --untared

# Custom window size
python -m yostlabs.tools.animate_sensor --width 800 --height 600
```

**Programmatic usage:**
```python
from yostlabs.tools.animate_sensor import animate_sensor

# Auto-discover sensor and show tared orientation
animate_sensor()

# Or provide your own sensor instance
from yostlabs.tss3.api import ThreespaceSensor
sensor = ThreespaceSensor()
animate_sensor(sensor=sensor, use_tared=True, window_width=800, window_height=600)
```

**Controls:**
- WASD + Space/Shift: Move camera
- Mouse drag (right button): Rotate camera
- ESC or close window: Exit

## Package Structure

- `yostlabs.graphics.core`: Core scene graph and rendering objects (GameObject, Scene, Camera, etc.)
- `yostlabs.graphics.prefabs`: Specialized visualization objects (TriAxesObject, LabeledTriAxesObject)
- `yostlabs.graphics.scene_prefabs`: Pre-configured scene templates (CameraScene, OrientationScene)
- `yostlabs.graphics.context`: OpenGL context management (GL_Context)
- `yostlabs.graphics.texture_renderer`: Framebuffer/texture rendering
- `yostlabs.graphics.font`: Font loading and text rendering
- `yostlabs.graphics.loaders`: Resource loaders (OBJ file parser)
- `yostlabs.graphics.resources`: Package resource utilities (fonts, models, shaders)
- `yostlabs.graphics.dpg`: DearPyGui backend integration
- `yostlabs.graphics.glfw`: GLFW backend specific components
