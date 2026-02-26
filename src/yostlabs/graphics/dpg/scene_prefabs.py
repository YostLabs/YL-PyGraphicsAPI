from yostlabs.graphics.texture_renderer import TextureRenderer
from yostlabs.graphics.core import SceneContext, Scene
from yostlabs.graphics.context import GL_Context

import dearpygui.dearpygui as dpg

import numpy as np

class DpgScene:
    """
    Wrapper around a Scene that adds DearPyGui texture rendering capabilities.
    Manages TextureRenderer and provides convenient methods for DPG texture updates.
    
    This is a composition-based wrapper that delegates all Scene operations to an
    internal Scene object, allowing it to wrap existing scenes without duplication.
    
    Example usage:
        # Create DPG scene with internal scene
        dpg_scene = DpgScene(800, 600, name="My Scene")
        
        # Or wrap an existing scene
        existing_scene = Scene("My Scene")
        dpg_scene = DpgScene(800, 600, scene=existing_scene)
        
        # Initialize DPG
        dpg.create_context()
        
        # Create DPG texture (after dpg.create_context())
        dpg_scene.createDpgTexture()
        
        # Setup scene (add camera, objects, etc.)
        dpg_scene.set_camera(camera)
        dpg_scene.add_child(model)
        
        # Render and update texture
        dpg_scene.render()
        dpg_scene.update_dpg_texture()  # Updates internal texture
        
        # Or update a custom texture
        dpg_scene.update_dpg_texture(my_custom_texture)
    """
    
    def __init__(self, texture_width: int, texture_height: int, scene: Scene = None, name: str = "DpgScene"):
        """
        Initialize a DPG scene with texture rendering capabilities.
        
        Args:
            texture_width: Width of the render texture
            texture_height: Height of the render texture
            scene: Optional Scene object to wrap. If None, creates a new Scene.
            name: Scene name (only used if scene is None)
        """
        # Create or use provided scene
        self.scene = scene if scene is not None else Scene(name)
        
        self.texture_width = texture_width
        self.texture_height = texture_height
        
        # Create texture renderer
        self.renderer = TextureRenderer(texture_width, texture_height)
        
        # DPG texture will be created later via createDpgTexture()
        self.dpg_raw_texture = None
        self._cached_texture_data = None
    
    def createDpgTexture(self):
        """
        Create the DPG texture registry and raw texture.
        Must be called after dpg.create_context().
        
        Returns:
            DPG texture ID
        """
        with dpg.texture_registry():
            self.dpg_raw_texture = dpg.add_raw_texture(
                width=self.texture_width,
                height=self.texture_height,
                default_value=np.zeros(self.texture_width * self.texture_height * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba
            )
        return self.dpg_raw_texture
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """
        Render the scene to the internal texture.
        Automatically handles the renderer context.
        
        Args:
            shader_program: OpenGL shader program to use. If None, uses default shader from GL_Context.
            parent_matrix: World transformation matrix from parent (for internal use)
            scene_context: Optional scene rendering context
        """

        with self.renderer:
            self.scene.render(shader_program=shader_program, parent_matrix=parent_matrix, scene_context=scene_context)
        
        # Invalidate texture data cache
        self._cached_texture_data = None
    
    def get_texture_data(self) -> np.ndarray:
        """
        Get the rendered texture data formatted for DearPyGui.
        Returns flipped and flattened texture data ready for dpg.set_value().
        
        Returns:
            Numpy array of texture data in DPG format
        """
        if self._cached_texture_data is None:
            raw_texture = self.renderer.get_texture_pixels()
            self._cached_texture_data = np.flip(raw_texture, 0).flatten()
        return self._cached_texture_data
    
    def update_dpg_texture(self, dpg_texture_id=None) -> None:
        """
        Update a DPG texture with the rendered scene.
        
        Args:
            dpg_texture_id: DearPyGui texture ID to update (optional).
                          If None, updates the internal dpg texture.
        """
        if dpg_texture_id is None:
            dpg_texture_id = self.dpg_raw_texture
        
        if dpg_texture_id is None:
            raise RuntimeError(
                "No DPG texture to update. Either call createDpgTexture() first "
                "or provide a dpg_texture_id parameter."
            )
        
        texture_data = self.get_texture_data()
        dpg.set_value(dpg_texture_id, texture_data)
    
    def get_dpg_texture(self):
        """
        Get the internal DPG raw texture created by createDpgTexture().
        
        Returns:
            DPG texture ID, or None if createDpgTexture() hasn't been called
        """
        return self.dpg_raw_texture
    
    def destroy(self) -> None:
        """Clean up resources including the texture renderer."""
        if self.renderer is not None:
            self.renderer.destroy()
            self.renderer = None
        self.scene.destroy()
    
    # Delegate all Scene methods to the wrapped scene
    def set_background_color(self, r: float, g: float, b: float, a: float = 1.0) -> None:
        """Set the background/clear color."""
        self.scene.set_background_color(r, g, b, a)
    
    def set_camera(self, camera) -> None:
        """Set the camera for this scene."""
        self.scene.set_camera(camera)
    
    def set_lighting(self, light_pos: tuple, view_pos: tuple, light_color: tuple) -> None:
        """Configure scene lighting."""
        self.scene.set_lighting(light_pos, view_pos, light_color)
    
    def add_child(self, child) -> None:
        """Attach a child game object to the scene."""
        self.scene.add_child(child)
    
    def remove_child(self, child) -> None:
        """Detach a child game object from the scene."""
        self.scene.remove_child(child)
    
    def find_child(self, name: str):
        """Find a child by name (recursive search)."""
        return self.scene.find_child(name)
    
    def set_active(self, active: bool) -> None:
        """Enable or disable the scene and its children."""
        self.scene.set_active(active)
    
    def set_visible(self, visible: bool) -> None:
        """Show or hide the scene (doesn't affect children)."""
        self.scene.set_visible(visible)
    
    def add_overlay(self, overlay) -> None:
        """Add a HUD overlay to the scene."""
        self.scene.add_overlay(overlay)
    
    def remove_overlay(self, overlay) -> None:
        """Remove a HUD overlay from the scene."""
        self.scene.remove_overlay(overlay)
    
    @property
    def name(self):
        """Get the scene name."""
        return self.scene.name
    
    @name.setter
    def name(self, value):
        """Set the scene name."""
        self.scene.name = value
    
    @property
    def camera(self):
        """Get the scene camera."""
        return self.scene.camera
    
    @camera.setter
    def camera(self, value):
        """Set the scene camera."""
        self.scene.camera = value
    
    @property
    def background_color(self):
        """Get the scene background color."""
        return self.scene.background_color
    
    @background_color.setter
    def background_color(self, value):
        """Set the scene background color."""
        self.scene.background_color = value
    
    @property
    def lighting_config(self):
        """Get the scene lighting configuration."""
        return self.scene.lighting_config
    
    @lighting_config.setter
    def lighting_config(self, value):
        """Set the scene lighting configuration."""
        self.scene.lighting_config = value
