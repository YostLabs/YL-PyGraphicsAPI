"""
Pre-built Scene classes with common configurations.
Contains Scene subclasses that are already partially set up for common use cases.
"""

from yostlabs.graphics.core import Scene, Camera, Font, GameObject, ModelObject, HudOverlay
from yostlabs.graphics.prefabs import LabeledTriAxesObject
from yostlabs.graphics.scene_components import CameraMover

import numpy as np
from yostlabs.math.axes import AxisOrder

class CameraScene(Scene):
    """
    A Scene that creates and configures its own camera on initialization.
    
    The camera is set up with the following defaults:
    - Position: (0, 0, 10) looking at origin
    - Perspective: 90° FOV, 0.01-1000.0 clipping planes
    - Viewport: Matches provided width and height
    
    Example usage:
        # Create scene with camera
        scene = CameraScene(800, 600, name="My Scene")
        
        # Camera is already set up and ready to use
        # Modify camera if needed
        scene.camera.set_position([0, 0, 100])
        scene.camera.set_perspective(60.0, 800/600, 0.1, 500.0)
    """
    
    def __init__(self, width: int, height: int, name: str = "CameraScene"):
        """
        Initialize a scene with a pre-configured camera.
        
        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
            name: Scene name
        """
        super().__init__(name)
        
        # Create and configure camera
        self.camera = Camera("Main Camera")
        self.camera.set_viewport(width, height)
        self.camera.set_perspective(90.0, width / height, 0.01, 1000.0)
        self.camera.set_position([0, 0, 10])
        self.camera.look_at(np.array([0, 0, 0.0], dtype=np.float32))
        
        # Set as the scene's camera
        self.set_camera(self.camera)


class OrientationScene(CameraScene):
    """
    A Scene with a model and axes for visualizing 3D orientation.
    
    Includes:
    - Pre-configured camera (from CameraScene)
    - Model object (provided by user)
    - Labeled tri-axes as child of model
    - Camera control methods for interactive navigation
    
    Example usage:
        # Create scene with model
        model = ModelObject("MyModel", "path/to/model.obj")
        scene = OrientationScene(800, 600, model, font=my_font)
        
        # In main loop, optionally call camera controls
        while running:
            scene.update_camera_pos()
            scene.update_camera_rotation()
            scene.render()
    """
    
    def __init__(self, width: int, height: int, model: GameObject, font: Font = None, name: str = "OrientationScene"):
        """
        Initialize an orientation scene with model and axes.
        
        Args:
            width: Viewport width in pixels
            height: Viewport height in pixels
            model: GameObject to add to the scene
            font: Font for axis labels (optional, can be None)
            name: Scene name
        """
        super().__init__(width, height, name)

        # Store viewport dimensions for camera controls
        self.viewport_width = width
        self.viewport_height = height

        # Core Scene Setup
        self.camera.set_position([0, 0, 3])
        self.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)

        # Add orientation indicator overlay in top-left corner
        self.orientation_indicator = OrientationIndicatorOverlay(x=0, y=height-130, font=font)
        self.add_overlay(self.orientation_indicator)

        # Add model to scene
        self.model = model
        self.add_child(model)

        # Create tri-axes as child of model
        self.axes = LabeledTriAxesObject("Axes", font=font)
        model.add_child(self.axes)

        #Automatically scale model objects to fit the space. If not a model object,
        #then user should call scale_model_to_view_space manually with an appropriate size for non-model objects.
        if isinstance(model, ModelObject):
            bmin, bmax = model.model.get_global_bounding_box()
            max_extent = max(max(abs(bmin)), max(abs(bmax)))
            self.scale_model_to_view_space(max_extent)

        # Scale axes inversely to model's scale to prevent model scaling from affecting the axes
        self.update_axes_scale()
    
        # Create camera mover for interactive controls (to be set by user to control inputs)
        self.camera_mover: CameraMover = None
        
        # Store initial camera state for reset
        self._initial_position = self.camera.position.copy()
        self._initial_rotation = self.camera.rotation.copy()
        
    def scale_model_to_view_space(self, base_model_max_extent=0.5, modify_original_scale=True):
        """
        Scales the model to fit inside a cube with sides of length 1 unit, centered at the origin. 
        The scaling is uniform and based on the largest dimension of the model's bounding box.
        This is done to ensure that any model can be properly visualized in the scene without needing 
        manual scaling by the user.
    
        The original scale will still be taken into account to allow the user to further
        scale the model. So, for example, if the scale is set to 1.5 before this function
        is called, then the scene will first uniformly scale the model so that its largest dimension
        is 1 unit, and then apply the original scale on top of that.
        """
        original_scale = self.model.scale
        NEW_MAX_EXTENT = 0.5
        self.model.set_scale(NEW_MAX_EXTENT / base_model_max_extent)
        if modify_original_scale:
            self.model.set_scale(self.model.scale * original_scale)  # Apply original scale on top of the new scale
        self.update_axes_scale()  # Update axes scale to maintain constant size after scaling the model


    def update_axes_scale(self):
        """
        Updates the 3D Axes scale to maintain a constant size. The scale
        is based on the parent model's scale, so that the axes will not be 
        affected by changes to the model's scale.
        """
        inverse_scale = 1.0 / self.model.scale
        self.axes.set_scale(inverse_scale)

    def update_camera_pos(self) -> None:
        """
        Update camera position based on keyboard input (WASD + Space/Shift).
        Call this in your main loop if you want keyboard camera controls.
        """
        self.camera_mover.update_camera_pos()
    
    def update_camera_rotation(self) -> None:
        """
        Update camera rotation based on mouse input (right-click drag).
        Call this in your main loop if you want mouse camera controls.
        """
        self.camera_mover.update_camera_rotation()

    def set_camera_speed(self, speed: float) -> None:
        """Set the camera movement speed (units per second)."""
        self.camera_mover.set_camera_speed(speed)
    
    def set_camera_mover(self, camera_mover: CameraMover) -> None:
        """
        Set the camera mover implementation.
        
        Args:
            camera_mover: A CameraMover instance (e.g., GlfwCameraMover, DpgCameraMover)
        """
        self.camera_mover = camera_mover
        self.camera_mover.set_camera_speed(1)
    
    def reset_camera(self) -> None:
        """
        Reset camera to its initial position and rotation.
        Also resets virtual mouse position to ensure interactive rotation state is cleared.
        """
        self.camera.position = self._initial_position.copy()
        self.camera.rotation = self._initial_rotation.copy()
        self.camera_mover.reset_virtual_mouse()
    
    def set_axis_order(self, order: str|AxisOrder) -> None:
        """
        Set the axis order for both main axes and orientation indicator.
        
        Args:
            order: List of 3 integers [0-2] specifying axis mapping.
                  order[i] = j means original axis i goes to position j.
        """
        self.axes.set_axis_order(order)
        self.orientation_indicator.set_axis_order(order)


class OrientationIndicatorOverlay(HudOverlay):
    """
    A HUD overlay showing tri-axes orientation indicator in the top-left corner.
    
    Supports dynamic axis remapping with automatic label positioning to ensure
    the forward/Z-axis label remains visible regardless of axis configuration.
    
    Example usage:
        # Create orientation indicator
        indicator = OrientationIndicatorOverlay(x=0, y=500, size=100, font=my_font)
        
        # Add to scene
        scene.add_overlay(indicator)
        
        # Optionally change axis order
        indicator.set_axis_order([2, 0, 1])  # X->Z, Y->X, Z->Y
    """
    
    def __init__(self, x: int, y: int, size: int = 130, font: Font = None):
        """
        Initialize the orientation indicator overlay.
        
        Args:
            x: X position of the overlay in screen coordinates
            y: Y position of the overlay in screen coordinates
            size: Size of the overlay square in pixels (default 130)
            font: Optional font for axis labels
        """
        # Create small scene for orientation indicator
        indicator_scene = Scene("OrientationIndicator")
        
        # Set up lighting for the indicator
        indicator_scene.set_lighting(
            light_pos=(20, 30, 25),
            view_pos=(35, 35, 35),
            light_color=(1.0, 1.0, 1.0)
        )

        # Create camera positioned to view axes from a nice angle
        self.camera = Camera("IndicatorCamera")
        self.camera.set_perspective(30.0, 1.0, 0.01, 150.0)  # Narrower FOV for less distortion
        self.camera.set_position([0, 0, 80/23])
        self.camera.look_at(np.array([0, 0, 0], dtype=np.float32))
        indicator_scene.set_camera(self.camera)
        
        # Create tri-axes
        self.axes = LabeledTriAxesObject("IdentityAxes", font=font)        
        # Store original label positions for restoration when axis order is changed
        self._original_label_positions = {
            'x': self.axes.x_label.position.copy(),
            'y': self.axes.y_label.position.copy(),
            'z': self.axes.z_label.position.copy()
        }
        
        self.axes.set_scale(0.7)
        indicator_scene.add_child(self.axes)
        
        # Initialize parent HudOverlay
        super().__init__(indicator_scene, x=x, y=y, width=size, height=size, auto_camera_viewport=False)
        
        self._update_axis_mapping()
    
    def set_axis_order(self, order: str|AxisOrder) -> None:
        """
        Set the axis order and update label positions for visibility.
        """
        # Apply axis order to the axes object
        self.axes.set_axis_order(order)
        self._update_axis_mapping()

    def _update_camera_viewport(self):
        #This is based on the HUD overlay size and position

        # Render will have a large amount of black space, so to make the object as big as possible
        # while fitting the given space on the HUD, create a larget texture and then crop. This
        # ratio was compute via experimentation to maximize visible area.
        x_pos, y_pos, display_width, display_height = self.viewport

        texture_width = int(display_width * 200 / 130)
        texture_height = int(display_height * 200 / 130)

        x_shift = -int(texture_width * 0.35)
        y_shift = -int(texture_height * 0.40)
        
        horizontal_axis_index = self.axes.axis_order.order.index(0)
        vertical_axis_index = self.axes.axis_order.order.index(1)

        if self.axes.axis_order.multipliers[horizontal_axis_index] < 0:
            x_shift = 0
        if self.axes.axis_order.multipliers[vertical_axis_index] < 0:
            y_shift = 0

        self.camera.set_viewport(
            width=texture_width,
            height=texture_height,
            x=x_pos + x_shift,
            y=y_pos + y_shift
        )

    def _update_axis_mapping(self):
        """Internal method to update axis mapping and label positions."""
        #Adjust label positions to ensure Forward label is visible
        
        # Restore all labels to their original positions
        self.axes.x_label.position = self._original_label_positions['x'].copy()
        self.axes.y_label.position = self._original_label_positions['y'].copy()
        self.axes.z_label.position = self._original_label_positions['z'].copy()
        
        # Find which original axis is mapped to position 0 (OpenGL +X / right side)
        # axis_order[i] = j means original axis i goes to position j
        horizontal_axis_index = self.axes.axis_order.order.index(0)
        vertical_axis_index = self.axes.axis_order.order.index(1)
        forward_axis_index = self.axes.axis_order.order.index(2)
        
        # Check the multiplier for that axis to know if it points right (+) or left (-)
        negate_shift = 1
        if self.axes.axis_order.multipliers[horizontal_axis_index] > 0: #The horizontal axis points right, so shift left (negative)
            negate_shift *= -1
        
        if self.axes.axis_order.multipliers[forward_axis_index] < 0:
            # Forward axis is inverted, so flip the shift direction
            negate_shift *= -1
        
        forward_arrow, forward_label = self.axes[forward_axis_index]
        label_shift = 0.217 * negate_shift

        # Ensure the forward label is at proper depth and shift
        forward_label.position[0] += label_shift
        forward_label.position[2] = 0

        #Now update the overlays x/y position to render only the desired quadrant.
        self._update_camera_viewport()
