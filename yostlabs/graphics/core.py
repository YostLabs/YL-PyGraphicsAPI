"""
Game Object system for 3D scene hierarchy and composition.
Provides a scene graph with parent-child relationships and local/world transforms.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Sequence

from OpenGL.GL import *
import ctypes

import yostlabs.math.quaternion as yl_quat
import numpy as np

from yostlabs.graphics.loaders.obj_loader import OBJ
from yostlabs.graphics.font import Font
from yostlabs.graphics.context import GL_Context

class SceneContext:
    """Contains rendering context information passed through the scene hierarchy."""
    
    def __init__(self, 
                 projection_matrix: np.ndarray = None,
                 view_matrix: np.ndarray = None,
                 light_pos: np.ndarray = None,
                 view_pos: np.ndarray = None,
                 light_color: np.ndarray = None):
        """Initialize scene rendering context.
        
        Args:
            projection_matrix: 4x4 projection matrix
            view_matrix: 4x4 view matrix
            light_pos: Light position (x, y, z)
            view_pos: View/camera position (x, y, z)
            light_color: Light color (r, g, b)
        """
        self.projection_matrix = projection_matrix if projection_matrix is not None else np.identity(4)
        self.view_matrix = view_matrix if view_matrix is not None else np.identity(4)
        self.light_pos = light_pos if light_pos is not None else np.array([40.0, 60.0, 50.0])
        self.view_pos = view_pos if view_pos is not None else np.array([0.0, 0.0, -80.0])
        self.light_color = light_color if light_color is not None else np.array([1.0, 1.0, 1.0])


class GameObject(ABC):
    """
    Base class for all game objects in the scene.
    Implements hierarchical transforms with local and world space coordinates.
    """
    
    def __init__(self, name: str = "GameObject"):
        """
        Initialize a GameObject.
        
        Args:
            name: Display name for this game object
        """
        self.name = name
        self.parent: Optional[GameObject] = None
        self.children: List[GameObject] = []
        
        # Local transform
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rotation = np.identity(3, dtype=np.float32)  # 3x3 rotation matrix
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        # Active state
        self.active = True # If false, object and children are not rendered
        self.visible = True # If false, object is not rendered but children may be
    
    def add_child(self, child: GameObject) -> None:
        """Attach a child game object to this parent."""
        if child.parent is not None:
            child.parent.children.remove(child)
        
        self.children.append(child)
        child.parent = self
    
    def remove_child(self, child: GameObject) -> None:
        """Detach a child game object."""
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def set_position(self, pos: Sequence[float]) -> None:
        """Set local position from a sequence (list, tuple, or array) of [x, y, z]."""
        self.position = np.array(pos, dtype=np.float32)
    
    def set_rotation_quat(self, quat: np.ndarray) -> None:
        """Set local rotation from quaternion (x,y,z,w)."""
        # Convert quaternion to 3x3 rotation matrix
        self.rotation = yl_quat.quaternion_to_3x3_rotation_matrix(quat)
    
    def set_rotation_matrix(self, matrix: np.ndarray) -> None:
        """Set local rotation from 3x3 matrix."""
        self.rotation = np.array(matrix, dtype=np.float32)

    def get_rotation_as_matrix(self) -> np.ndarray:
        """Get local rotation as a 3x3 matrix."""
        return self.rotation
    
    def get_rotation_as_quat(self):
        """Get local rotation as a quaternion (x, y, z, w)."""
        return yl_quat.quaternion_from_3x3_rotation_matrix(self.rotation)
    
    def set_scale(self, scale: Union[float, Sequence[float]]) -> None:
        """
        Set local scale.
        
        Args:
            scale: Either a single scalar (applied to all axes) or a sequence of [x, y, z]
        """
        try:
            # Check if it's a sequence by trying to get length
            len(scale)
            # It's a sequence, use as-is
            self.scale = np.array(scale, dtype=np.float32)
        except TypeError:
            # It's a scalar (int, float, np.float, etc.), apply to all axes
            self.scale = np.array([scale, scale, scale], dtype=np.float32)
    
    def get_local_matrix(self) -> np.ndarray:
        """Get the local transformation matrix (4x4)."""
        matrix = np.identity(4)
        
        # Apply scale first (creates a diagonal scale matrix)
        matrix[0, 0] = self.scale[0]
        matrix[1, 1] = self.scale[1]
        matrix[2, 2] = self.scale[2]
        
        # Apply rotation (multiply rotation matrix with scale matrix: R * S)
        # This preserves the rotation's orthogonality while scaling
        matrix[:3, :3] = self.rotation @ matrix[:3, :3]
        
        # Apply position (translation)
        matrix[0, 3] = self.position[0]
        matrix[1, 3] = self.position[1]
        matrix[2, 3] = self.position[2]
        
        return matrix
    
    def get_world_matrix(self) -> np.ndarray:
        """Get the world transformation matrix (4x4)."""
        local = self.get_local_matrix()
        
        if self.parent is not None:
            parent_world = self.parent.get_world_matrix()
            return parent_world @ local
        
        return local
    
    def set_active(self, active: bool) -> None:
        """Enable or disable this game object and its children."""
        self.active = active
    
    def set_visible(self, visible: bool) -> None:
        """Show or hide this game object (doesn't affect children)."""
        self.visible = visible
    
    
    @abstractmethod
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """
        Render this game object.
        
        Args:
            shader_program: OpenGL shader program to use. If None, use default shader from GL_Context.
            parent_matrix: World transformation matrix from parent (for internal use)
            scene_context: Optional scene rendering context containing projection, view, and lighting info
        """
        pass
    
    def render_tree(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Recursively render this object and all active children.
        
        Args:
            shader_program: OpenGL shader program to use. If None, use default shader from GL_Context.
            parent_matrix: World transformation matrix from parent
            scene_context: Scene rendering context to pass down the hierarchy
        """
        if not self.active:
            return
        
        if shader_program is None:
            shader_program = GL_Context.get_shader_program()

        # Get world matrix
        local = self.get_local_matrix()
        if parent_matrix is not None:
            world_matrix = parent_matrix @ local
        else:
            world_matrix = local
        
        # Render this object
        if self.visible:
            self.render(shader_program=shader_program, parent_matrix=world_matrix, scene_context=scene_context)
        
        # Render children
        for child in self.children:
            child.render_tree(shader_program=shader_program, parent_matrix=world_matrix, scene_context=scene_context)
    
    def find_child(self, name: str) -> Optional[GameObject]:
        """Find a child by name (recursive search)."""
        for child in self.children:
            if child.name == name:
                return child
            found = child.find_child(name)
            if found is not None:
                return found
        return None
    
    def destroy(self) -> None:
        """Clean up resources."""
        for child in self.children[:]:
            child.destroy()
        self.children.clear()
        if self.parent is not None:
            self.parent.remove_child(self)


class TransformNode(GameObject):
    """
    A game object that only represents a transform in space.
    Useful for grouping and organizing object hierarchies.
    """
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Transform nodes don't render anything themselves."""
        pass


class Camera(GameObject):
    """
    A camera object that manages view and projection matrices.
    Uses rotation (quaternion) to determine camera orientation.
    Default orientation: forward=[0,0,-1], up=[0,1,0], right=[1,0,0]
    """
    
    def __init__(self, name: str = "Camera"):
        """
        Initialize a camera.
        
        Args:
            name: Object name
        """
        super().__init__(name)
        
        # Camera parameters
        self.fov = 45.0  # Field of view in degrees
        self.aspect_ratio = 1.0
        self.near_plane = 0.1
        self.far_plane = 1000.0
        
        # Viewport dimensions and position
        self.viewport_x = 0
        self.viewport_y = 0
        self.viewport_width = 800
        self.viewport_height = 600
    
    def set_viewport(self, width: int, height: int, x: int = 0, y: int = 0) -> None:
        """
        Set the viewport dimensions and position.
        
        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
            x: Viewport x position in pixels (default 0)
            y: Viewport y position in pixels (default 0)
        """
        self.viewport_x = x
        self.viewport_y = y
        self.viewport_width = width
        self.viewport_height = height
        self.aspect_ratio = width / height if height > 0 else 1.0
    
    def apply_viewport(self) -> None:
        """Apply this camera's viewport to OpenGL."""
        glViewport(self.viewport_x, self.viewport_y, self.viewport_width, self.viewport_height)
    
    def set_perspective(self, fov: float = None, aspect_ratio: float = None, near: float = None, far: float = None) -> None:
        """
        Set camera perspective parameters.
        
        Args:
            fov: Field of view in degrees
            aspect_ratio: Aspect ratio (width/height)
            near: Near clipping plane
            far: Far clipping plane
        """
        if fov is not None:
            self.fov = fov
        if aspect_ratio is not None:
            self.aspect_ratio = aspect_ratio
        if near is not None:
            self.near_plane = near
        if far is not None:
            self.far_plane = far
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get the projection matrix."""
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        projection = np.zeros((4, 4), dtype=np.float32)
        projection[0, 0] = f / self.aspect_ratio
        projection[1, 1] = f
        projection[2, 2] = (self.far_plane + self.near_plane) / (self.near_plane - self.far_plane)
        projection[2, 3] = (2.0 * self.far_plane * self.near_plane) / (self.near_plane - self.far_plane)
        projection[3, 2] = -1.0
        
        return projection
    
    def get_view_matrix(self) -> np.ndarray:
        """Get the view matrix from camera position and rotation.
        
        The view matrix is the inverse of the world matrix.
        For rigid body transforms: view = [R^T | -R^T * t]
        """
        # Get world matrix
        world_matrix = self.get_world_matrix()
        
        # Extract rotation (3x3) and translation (3x1)
        rotation = world_matrix[:3, :3]
        translation = world_matrix[:3, 3]
        
        # View matrix = inverse of world matrix
        # For rigid transforms: transpose rotation and negate translated position
        view = np.identity(4, dtype=np.float32)
        view[:3, :3] = rotation.T
        view[:3, 3] = -rotation.T @ translation
        
        return view
    
    def look_at(self, target: np.ndarray) -> None:
        """
        Set camera rotation to look at a target point.
        
        Args:
            target: Position in world space to look at
        """
        # Compute forward vector (from eye to target)
        forward = target - self.position
        forward = forward / (np.linalg.norm(forward) + 1e-6)
        
        # World up vector
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Compute right vector (perpendicular to forward and up)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-6)
        
        # Recompute up vector to ensure orthonormal basis
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-6)
        
        # Create 3x3 rotation matrix from basis vectors
        # The columns of this matrix are right, up, -forward
        rotation_matrix = np.identity(3, dtype=np.float32)
        rotation_matrix[:, 0] = right
        rotation_matrix[:, 1] = up
        rotation_matrix[:, 2] = -forward
        
        # Convert rotation matrix to quaternion and set rotation
        self.set_rotation_matrix(rotation_matrix)
    
    def get_right(self) -> np.ndarray:
        """Get the camera's right vector in world space."""
        world_matrix = self.get_world_matrix()
        return world_matrix[:3, 0]
    
    def get_up(self) -> np.ndarray:
        """Get the camera's up vector in world space."""
        world_matrix = self.get_world_matrix()
        return world_matrix[:3, 1]
    
    def get_forward(self) -> np.ndarray:
        """Get the camera's forward vector in world space (direction into the screen)."""
        world_matrix = self.get_world_matrix()
        return -world_matrix[:3, 2]
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Cameras don't render anything themselves."""
        pass


class ModelObject(GameObject):
    """
    Game object that renders a 3D model (OBJ).
    Implements model caching so that multiple instances of the same model
    don't reload the geometry.
    """
    
    # Class-level cache for loaded models
    _model_cache: Dict[str, OBJ] = {}
    
    def __init__(self, name: str = "Model", model_path: str = None):
        """
        Initialize model object.
        
        Args:
            name: Object name
            model_path: Path to the OBJ model file
        """
        super().__init__(name)
        self.model: Optional[OBJ] = None
        self.model_path = model_path
        self.model_alpha = 1.0  # Alpha transparency (0.0-1.0)
        
        if model_path is not None:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load a model from file, using cache if available.
        
        Args:
            model_path: Path to the OBJ model file
        """
        self.model_path = model_path
        
        if model_path not in ModelObject._model_cache:
            # Load the model and cache it
            obj = OBJ(model_path)
            obj.generate()
            ModelObject._model_cache[model_path] = obj
        
        # Use cached model
        self.model = ModelObject._model_cache[model_path]
    
    def set_model(self, model: OBJ) -> None:
        """Set the OBJ model to render directly."""
        self.model = model
    
    def set_alpha(self, alpha: float) -> None:
        """Set the model alpha/transparency (0.0=transparent to 1.0=opaque)."""
        self.model_alpha = max(0.0, min(1.0, alpha))
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Render the 3D model."""
        if self.model is None:
            return
        
        if parent_matrix is None:
            parent_matrix = np.identity(4)
        
        if shader_program is None:
            shader_program = GL_Context.get_shader_program()
        glUseProgram(shader_program)
        
        # Disable depth writes for transparent objects
        if self.model_alpha < 1.0:
            glDepthMask(GL_FALSE)
        
        # Set model matrix uniform
        model_loc = glGetUniformLocation(shader_program, "model")
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, parent_matrix)
        
        # Set model alpha uniform
        alpha_loc = glGetUniformLocation(shader_program, "modelAlpha")
        glUniform1f(alpha_loc, self.model_alpha)
        
        # Render the model
        self.model.render()
        
        # Re-enable depth writes
        if self.model_alpha < 1.0:
            glDepthMask(GL_TRUE)


class TextMesh(GameObject):
    """
    Game object that renders text in world space.
    Uses a billboard shader so text always faces the camera.
    Uses the global TextRenderer from GL_Context.
    """
    
    def __init__(self, text: str = "", font: Font = None, name: str = "TextMesh",
                 text_size: float = 1.0, color: tuple = (1.0, 1.0, 1.0)):
        """
        Initialize a text mesh.
        
        Args:
            name: Object name
            font: Font object to use for rendering
            text: The text string to display
            text_size: Height of text in world units (default 1.0)
            color: RGB color tuple (0-1 range)
        """
        super().__init__(name)
        self.font = font
        self.text = text
        self.text_size = text_size  # Text size in world units
        self.color = np.array(color, dtype=np.float32)
    
    def set_font(self, font: Font) -> None:
        """Set the font for rendering."""
        self.font = font
    
    def set_text(self, text: str) -> None:
        """Set the text to display."""
        self.text = text
    
    def set_text_size(self, text_size: float) -> None:
        """Set the text size in world units."""
        self.text_size = text_size
    
    def set_color(self, r: float, g: float, b: float) -> None:
        """Set the text color."""
        self.color = np.array([r, g, b], dtype=np.float32)
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Render the text in world space using the global TextRenderer."""
        if self.font is None or self.text == "":
            return
        
        if GL_Context.text_renderer is None:
            return

        if parent_matrix is None:
            parent_matrix = np.identity(4)
        
        # Extract scale from parent matrix
        # Scale factors are the lengths of the rotation-scale matrix columns
        scale_x = np.linalg.norm(parent_matrix[:3, 0])
        scale_y = np.linalg.norm(parent_matrix[:3, 1])
        scale_z = np.linalg.norm(parent_matrix[:3, 2])
        # Use average scale for uniform scaling
        parent_scale = (scale_x + scale_y + scale_z) / 3.0
        
        # Render using the global text renderer
        GL_Context.text_renderer.render_text(
            self.font,
            self.text,
            0.0,  # x position (relative to world position)
            0.0,  # y position (relative to world position)
            self.text_size * parent_scale,
            tuple(self.color),
            model_matrix=parent_matrix,
            centered=True
        )


class Scene(GameObject):
    """
    A scene is the root container for all game objects.
    Manages rendering of the entire object hierarchy.
    """
    
    def __init__(self, name: str = "Scene"):
        """Initialize a scene."""
        super().__init__(name)
        self.camera: Optional[Camera] = None
        self.background_color = np.array([0, 0, 0, 0], dtype=np.float32)
        #self.background_color = np.array([0.41176474, 0.41176474, 0.41176474, 1.0], dtype=np.float32)
        self.lighting_config = {
            'lightPos': np.array([40.0, 60.0, 50.0]),
            'viewPos': np.array([0.0, 0.0, -80.0]),
            'lightColor': np.array([1.0, 1.0, 1.0])
        }
        # HUD overlays rendered after main scene
        self.overlays: List['HudOverlay'] = []
    
    def set_background_color(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Set the background/clear color."""
        self.background_color = np.array([r, g, b, a], dtype=np.float32)
    
    def set_camera(self, camera: Camera) -> None:
        """Set the camera for this scene."""
        self.camera = camera
    
    def set_lighting(self, light_pos: tuple, view_pos: tuple, light_color: tuple) -> None:
        """Configure scene lighting."""
        self.lighting_config['lightPos'] = np.array(light_pos)
        self.lighting_config['viewPos'] = np.array(view_pos)
        self.lighting_config['lightColor'] = np.array(light_color)
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None, clear_bits: int = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) -> None:
        """
        Render the entire scene.
        
        Args:
            shader_program: OpenGL shader program to use. If None, use default shader from GL_Context.
            parent_matrix: Parent transformation matrix (for internal use)
            scene_context: Scene rendering context
            clear_bits: OpenGL clear bits (default: GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                       Use GL_DEPTH_BUFFER_BIT for transparent overlays
        """
        
        # Apply camera viewport
        if self.camera is not None:
            self.camera.apply_viewport()
        
        if shader_program is None:
            shader_program = GL_Context.get_shader_program()

        # Clear with background color
        glClearColor(*self.background_color)
        glClear(clear_bits)
        
        glUseProgram(shader_program)
        
        # Create scene context with projection and view matrices
        if self.camera is not None:
            projection_matrix = self.camera.get_projection_matrix()
            view_matrix = self.camera.get_view_matrix()
            # Use actual camera position for lighting calculations
            camera_world_pos = self.camera.get_world_matrix()[:3, 3]
        else:
            projection_matrix = np.identity(4)
            view_matrix = np.identity(4)
            # Fall back to lighting_config value if no camera
            camera_world_pos = self.lighting_config['viewPos']
        
        # Create the scene context
        context = SceneContext(
            projection_matrix=projection_matrix,
            view_matrix=view_matrix,
            light_pos=self.lighting_config['lightPos'],
            view_pos=camera_world_pos,
            light_color=self.lighting_config['lightColor']
        )
        
        # Setup matrices on main shader
        projection_loc = glGetUniformLocation(shader_program, "projection")
        view_loc = glGetUniformLocation(shader_program, "view")
        
        glUniformMatrix4fv(projection_loc, 1, GL_TRUE, projection_matrix)
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, view_matrix)
        
        # Set matrices on text renderer if it exists
        if GL_Context.text_renderer is not None:
            GL_Context.text_renderer.set_projection_matrix(projection_matrix)
            GL_Context.text_renderer.set_view_matrix(view_matrix)
        
        # Setup lighting
        light_pos_loc = glGetUniformLocation(shader_program, "lightPos")
        glUniform3fv(light_pos_loc, 1, self.lighting_config['lightPos'])
        
        view_pos_loc = glGetUniformLocation(shader_program, "viewPos")
        glUniform3fv(view_pos_loc, 1, camera_world_pos)
        
        light_color_loc = glGetUniformLocation(shader_program, "lightColor")
        glUniform3fv(light_color_loc, 1, self.lighting_config['lightColor'])
        
        # Render all children with scene context
        for child in self.children:
            child.render_tree(shader_program, np.identity(4), scene_context=context)
        
        # Render overlays after main scene
        for overlay in self.overlays:
            overlay.render(shader_program=shader_program)
    
    def add_overlay(self, overlay: 'HudOverlay') -> None:
        """
        Add a HUD overlay to be rendered after the main scene.
        
        Args:
            overlay: HudOverlay to add
        """
        self.overlays.append(overlay)
    
    def remove_overlay(self, overlay: 'HudOverlay') -> None:
        """
        Remove a HUD overlay.
        
        Args:
            overlay: HudOverlay to remove
        """
        if overlay in self.overlays:
            self.overlays.remove(overlay)


class HudOverlay:
    """
    A viewport-based overlay that renders a scene to a specific region of the screen.
    
    This is useful for UI elements like orientation indicators, minimaps, etc.
    The overlay is rendered after the main scene and appears on top.
    
    Example usage:
        # Create overlay scene
        overlay_scene = Scene("Overlay")
        overlay_scene.set_camera(camera)
        overlay_scene.add_child(axes)
        
        # Create HUD overlay in top-left corner (100x100 pixels)
        hud = HudOverlay(overlay_scene, x=0, y=500, width=100, height=100)
        
        # Render overlay after main scene
        hud.render()
    """
    
    def __init__(self, scene: Scene, x: int, y: int, width: int, height: int, auto_camera_viewport=True):
        """
        Initialize a HUD overlay.
        
        Args:
            scene: Scene to render in the overlay region
            x: X position of overlay (pixels from left)
            y: Y position of overlay (pixels from bottom)
            width: Width of overlay in pixels
            height: Height of overlay in pixels
            auto_camera_viewport: Whether to automatically set the camera viewport to match the overlay (default: True)
        """
        self.scene = scene
        self.viewport = [x, y, width, height]

        # Configure camera viewport to match HUD if possible
        if auto_camera_viewport and self.scene.camera is not None:
            self.scene.camera.set_viewport(
                width=self.viewport[2],
                height=self.viewport[3],
                x=self.viewport[0],
                y=self.viewport[1]
            )
    
    def render(self, shader_program: int = None) -> None:
        """
        Render the overlay scene to its viewport region.
        
        Args:
            shader_program: OpenGL shader program to use. If None, use default shader from GL_Context.
        """
        
        # Enable scissor test to clip rendering to overlay region
        glEnable(GL_SCISSOR_TEST)
        glScissor(*self.viewport)
        
        # Render the scene, only clearing depth buffer (not color) for transparency
        clear_bits = GL_DEPTH_BUFFER_BIT
        if self.scene.background_color[3] == 1.0:
            # Opaque background, clear color buffer as well
            clear_bits |= GL_COLOR_BUFFER_BIT
        self.scene.render(shader_program=shader_program, clear_bits=clear_bits)
        
        # Restore state
        glDisable(GL_SCISSOR_TEST)
    
    def set_position(self, x: int, y: int) -> None:
        """Update the overlay position."""
        self.viewport = (x, y, self.viewport[2], self.viewport[3])
    
    def set_size(self, width: int, height: int) -> None:
        """Update the overlay size."""
        self.viewport = (self.viewport[0], self.viewport[1], width, height)


class ArrowObject(GameObject):
    """
    Game object that renders a 3D arrow using geometry shaders.
    The arrow is generated procedurally from a single point.
    
    The arrow consists of a rectangular shaft (stick) and a pyramidal tip (triangle).
    Color can be changed at runtime via uniform, making it efficient for dynamic coloring.
    
    Example usage:
        # Create a red arrow
        arrow = ArrowObject("MyArrow", 
                           stick_width=0.03, 
                           stick_length=0.75,
                           triangle_width=0.15, 
                           triangle_length=0.25,
                           arrow_color=(1.0, 0.0, 0.0))
        
        # Add to scene
        scene.add_child(arrow)
        
        # Change color at runtime
        arrow.set_arrow_color(0.0, 1.0, 0.0)  # Now green
        
        # Update arrow dimensions
        arrow.set_arrow_parameters(stick_length=20.0, triangle_width=4.0)
    """
    
    def __init__(self, name: str = "Arrow",
                 stick_width: float = 0.03,
                 stick_length: float = 0.75,
                 triangle_width: float = 0.15,
                 triangle_length: float = 0.25,
                 arrow_color: tuple = (1.0, 1.0, 1.0)):
        """
        Initialize an arrow object.
        Default total length of 1.
        
        Args:
            name: Object name
            stick_width: Width of the arrow shaft
            stick_length: Length of the arrow shaft
            triangle_width: Width of the arrow head base
            triangle_length: Length of the arrow head
            arrow_color: RGB color tuple (0.0-1.0)
        """
        super().__init__(name)
        
        # Arrow parameters
        self.stick_width = stick_width
        self.stick_length = stick_length
        self.triangle_width = triangle_width
        self.triangle_length = triangle_length
        # Store color as RGBA
        self.arrow_color = np.array([arrow_color[0], arrow_color[1], arrow_color[2], 1.0], dtype=np.float32)
        
        # Create a simple VAO with a single point at origin
        self.vao = self._create_point_vao()
    
    def _create_point_vao(self):
        """Create a VAO containing a single point at the origin."""
        
        vao = glGenVertexArrays(1)
        vbo = glGenBuffers(1)
        
        glBindVertexArray(vao)
        
        # Single point at origin
        vertex_data = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
        
        # Position attribute
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        
        glBindVertexArray(0)
        
        return vao
    
    def set_arrow_color(self, r: float, g: float, b: float) -> None:
        """Set the arrow color (RGB values 0.0-1.0)."""
        self.arrow_color[0] = r
        self.arrow_color[1] = g
        self.arrow_color[2] = b
    
    def set_arrow_alpha(self, alpha: float) -> None:
        """Set the arrow alpha/transparency (0.0=transparent to 1.0=opaque)."""
        self.arrow_color[3] = max(0.0, min(1.0, alpha))
    
    def set_arrow_parameters(self, stick_width: float = None, stick_length: float = None,
                            triangle_width: float = None, triangle_length: float = None):
        """Update arrow geometry parameters."""
        if stick_width is not None:
            self.stick_width = stick_width
        if stick_length is not None:
            self.stick_length = stick_length
        if triangle_width is not None:
            self.triangle_width = triangle_width
        if triangle_length is not None:
            self.triangle_length = triangle_length
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """Render the arrow using the geometry shader.
        
        Args:
            shader_program: OpenGL shader program (not used, arrow has its own)
            parent_matrix: World transformation matrix
            scene_context: Scene rendering context (required for arrow rendering)
        """
        if parent_matrix is None:
            parent_matrix = np.identity(4)
        
        # Validate scene context is provided
        if scene_context is None:
            raise ValueError(f"ArrowObject '{self.name}' requires scene_context for rendering")
        
        # Use arrow shader program
        arrow_shader = GL_Context.get_arrow_shader_program()
        glUseProgram(arrow_shader)
        
        # Set projection and view matrices from scene context
        proj_loc = glGetUniformLocation(arrow_shader, "projection")
        glUniformMatrix4fv(proj_loc, 1, GL_TRUE, scene_context.projection_matrix)
        
        view_loc = glGetUniformLocation(arrow_shader, "view")
        glUniformMatrix4fv(view_loc, 1, GL_TRUE, scene_context.view_matrix)
        
        # Set model matrix
        model_loc = glGetUniformLocation(arrow_shader, "model")
        glUniformMatrix4fv(model_loc, 1, GL_TRUE, parent_matrix)
        
        # Set arrow parameters
        glUniform1f(glGetUniformLocation(arrow_shader, "stickWidth"), self.stick_width)
        glUniform1f(glGetUniformLocation(arrow_shader, "stickLength"), self.stick_length)
        glUniform1f(glGetUniformLocation(arrow_shader, "triangleWidth"), self.triangle_width)
        glUniform1f(glGetUniformLocation(arrow_shader, "triangleLength"), self.triangle_length)
        
        # Set arrow color (RGBA)
        color_loc = glGetUniformLocation(arrow_shader, "arrowColor")
        glUniform4fv(color_loc, 1, self.arrow_color)
        
        # Set lighting uniforms from scene context
        light_pos_loc = glGetUniformLocation(arrow_shader, "lightPos")
        glUniform3fv(light_pos_loc, 1, scene_context.light_pos)
        
        view_pos_loc = glGetUniformLocation(arrow_shader, "viewPos")
        glUniform3fv(view_pos_loc, 1, scene_context.view_pos)
        
        light_color_loc = glGetUniformLocation(arrow_shader, "lightColor")
        glUniform3fv(light_color_loc, 1, scene_context.light_color)
        
        # Draw the point (geometry shader will generate the arrow)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_POINTS, 0, 1)
        glBindVertexArray(0)
    
    def destroy(self) -> None:
        """Clean up OpenGL resources."""
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = None
        super().destroy()
