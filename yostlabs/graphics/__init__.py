"""
yl_graphics: 3D graphics engine with OpenGL scene graph and rendering primitives.

This package provides a complete 3D rendering system with hierarchical scene graph,
custom rendering objects, pre-built scene templates, and DearPyGui integration.
"""

__version__ = "0.1.0"

# Core scene graph components
from .core import (
    GameObject,
    TransformNode,
    Camera,
    Scene,
    ModelObject,
    TextMesh,
    ArrowObject,
    HudOverlay
)

from .loaders.obj_loader import OBJ

# Custom visualization objects
from . import prefabs

# Pre-built scene templates
from . import scene_prefabs

# OpenGL utilities
from .context import GL_Context, GL_AXIS_ORDER
from .texture_renderer import TextureRenderer
from .font import Font, TextRenderer

# Resource utilities
from . import resources

