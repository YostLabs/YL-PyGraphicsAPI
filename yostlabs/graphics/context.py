import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders

from yostlabs.graphics.font import TextRenderer, Font
from yostlabs.graphics.resources import get_font_path
from yostlabs.math.axes import AxisOrder

#Store how openGL maps to yostlabs sensor space
#OpenGL defaults to right handed with -Z into the screen
#The defaults axis order of a sensor is left-handed XYZ with +Z being forward/into the screen
#If the axis order of the sensor was set to XY-Z, no additional quat modification would be necessary.
GL_AXIS_ORDER = AxisOrder("xy-z")

class GL_Context:
    """OpenGL 3.3+ context"""
    initialized = False
    shader_program = None
    texture_shader_program = None
    arrow_shader_program = None
    texture_quad_vao = None
    text_renderer = None
    default_font = None
    window = None  # Store GLFW window reference

    @classmethod
    def init(cls, window_width: int = 200, window_height: int = 200, window_title: str = "OpenGL Renderer", visible: bool = False):
        """
        Initialize the OpenGL renderer with GLFW window.
        
        Args:
            window_width: Width of the window in pixels
            window_height: Height of the window in pixels
            window_title: Title of the window
            visible: Whether the window should be visible. Set true to render to an opengl window, false for DPG texture rendering only.
        """
        if cls.initialized: return
        if not glfw.init():
            print("Failed to init Glfw")
            return
        
        # Request OpenGL 3.3 core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE) #For MacOS compatibility
        
        cls.window = glfw.create_window(window_width, window_height, window_title, None, None)
        if not cls.window:
            glfw.terminate()
            print("Glfw window can't be created")
            exit()
        glfw.make_context_current(cls.window)

        # Load shader programs
        cls._load_shaders()
        cls._load_texture_shaders()
        cls._load_arrow_shaders()
        cls._create_texture_quad()
        
        # Create text renderer
        cls.text_renderer = TextRenderer()
        # Try loading fonts in order of preference
        cls.default_font = None
        for font_name in ['arial', 'FiraCode-Regular']: #Try a common system font, then a bundled font
            try:
                cls.default_font = Font(get_font_path(font_name))
                break
            except:
                continue
            

        # Setup modern OpenGL state
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glClearColor(105/255, 105/255, 105/255, 1.0)

        cls.initialized = True

    @classmethod
    def _load_shaders(cls):
        """Load and compile shaders from files"""
        from .resources import get_shader_path
        
        # Read shader source files
        with open(get_shader_path("vertex.glsl"), "r") as f:
            vertex_source = f.read()
        with open(get_shader_path("fragment.glsl"), "r") as f:
            fragment_source = f.read()
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        
        # Link program
        cls.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

    @classmethod
    def _load_texture_shaders(cls):
        """Load and compile texture display shaders"""
        from .resources import get_shader_path
        
        # Read shader source files
        with open(get_shader_path("texture_vertex.glsl"), "r") as f:
            vertex_source = f.read()
        with open(get_shader_path("texture_fragment.glsl"), "r") as f:
            fragment_source = f.read()
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        
        # Link program
        cls.texture_shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

    @classmethod
    def _load_arrow_shaders(cls):
        """Load and compile arrow shaders with geometry shader"""
        from .resources import get_shader_path
        
        # Read shader source files
        with open(get_shader_path("arrow_vertex.glsl"), "r") as f:
            vertex_source = f.read()
        with open(get_shader_path("arrow_geometry.glsl"), "r") as f:
            geometry_source = f.read()
        with open(get_shader_path("arrow_fragment.glsl"), "r") as f:
            fragment_source = f.read()
        
        # Compile shaders
        vertex_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        geometry_shader = shaders.compileShader(geometry_source, GL_GEOMETRY_SHADER)
        fragment_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        
        # Link program
        cls.arrow_shader_program = shaders.compileProgram(vertex_shader, geometry_shader, fragment_shader)

    @classmethod
    def _create_texture_quad(cls):
        """Create a quad VAO for rendering textures"""
        import numpy as np
        import ctypes
        
        # Full screen quad vertices and texcoords
        vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,  # bottom-left
             1.0, -1.0, 1.0, 0.0,  # bottom-right
             1.0,  1.0, 1.0, 1.0,  # top-right
            -1.0,  1.0, 0.0, 1.0   # top-left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        cls.texture_quad_vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        ebo = glGenBuffers(1)
        
        glBindVertexArray(cls.texture_quad_vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(0))
        
        # TexCoord attribute
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, ctypes.c_void_p(8))
        
        glBindVertexArray(0)

    @classmethod
    def get_shader_program(cls):
        """Get the active shader program"""
        return cls.shader_program

    @classmethod
    def get_texture_shader_program(cls):
        """Get the texture shader program"""
        return cls.texture_shader_program

    @classmethod
    def get_texture_quad_vao(cls):
        """Get the texture quad VAO"""
        return cls.texture_quad_vao

    @classmethod
    def get_arrow_shader_program(cls):
        """Get the arrow shader program"""
        return cls.arrow_shader_program

    @classmethod
    def get_window(cls):
        """Get the GLFW window"""
        return cls.window

    @classmethod
    def cleanup(cls):
        if cls.shader_program is not None:
            glDeleteProgram(cls.shader_program)
        if cls.window is not None:
            glfw.destroy_window(cls.window)
        glfw.terminate()

    
        