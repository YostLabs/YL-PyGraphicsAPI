"""
Custom game objects for common 3D visualization needs.
Contains specialized GameObject subclasses for axes, arrows, and other utilities.
"""

from __future__ import annotations
import numpy as np
from yostlabs.math.axes import AxisOrder
from yostlabs.graphics.core import GameObject, ArrowObject, TextMesh, Font, SceneContext


class TriAxesObject(GameObject):
    """
    Game object that renders three orthogonal arrows representing X, Y, Z axes.
    The arrows are colored red (X), green (Y), and blue (Z).
    
    Provides convenient methods to configure all arrows at once while still
    allowing individual arrow access and customization.
    
    Example usage:
        # Create tri-axes with custom arrow dimensions
        axes = TriAxesObject("Axes", 
                            stick_width=0.03, 
                            stick_length=0.75,
                            triangle_width=0.15, 
                            triangle_length=0.25)
        
        # Add to scene
        scene.add_child(axes)
        
        # Configure all arrows at once
        axes.set_all_arrow_parameters(stick_length=20.0)
        
        # Access individual arrows
        axes.x_arrow.set_arrow_color(1.0, 0.5, 0.5)  # Lighter red
    """
    
    def __init__(self, name: str = "TriAxes",
                 stick_width: float = 0.03,
                 stick_length: float = 0.75,
                 triangle_width: float = 0.15,
                 triangle_length: float = 0.25):
        """
        Initialize tri-axes object with three colored arrows.
        
        Args:
            name: Object name
            stick_width: Width of the arrow shaft for all arrows
            stick_length: Length of the arrow shaft for all arrows
            triangle_width: Width of the arrow head base for all arrows
            triangle_length: Length of the arrow head for all arrows
        """
        super().__init__(name)
        
        # Axis remapping configuration
        self.axis_order = AxisOrder("xyz")
        
        # Create X axis arrow (Red) - points along +X (rotated -90° around Y)
        self.x_arrow = ArrowObject(
            f"{name}_X",
            stick_width=stick_width,
            stick_length=stick_length,
            triangle_width=triangle_width,
            triangle_length=triangle_length,
            arrow_color=(1.0, 0.0, 0.0)  # Red
        )
        # Rotate to point along +X axis
        angle_rad = np.radians(-90)
        K = np.array([[0, -0, 1], [0, 0, -0], [-1, 0, 0]])
        R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        self.x_arrow.set_rotation_matrix(R)
        self.add_child(self.x_arrow)
        
        # Create Y axis arrow (Green) - points along +Y (rotated 90° around X)
        self.y_arrow = ArrowObject(
            f"{name}_Y",
            stick_width=stick_width,
            stick_length=stick_length,
            triangle_width=triangle_width,
            triangle_length=triangle_length,
            arrow_color=(0.0, 1.0, 0.0)  # Green
        )
        # Rotate to point along +Y axis
        angle_rad = np.radians(90)
        K = np.array([[0, -0, 0], [0, 0, -1], [0, 1, 0]])
        R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        self.y_arrow.set_rotation_matrix(R)
        self.add_child(self.y_arrow)
        
        # Create Z axis arrow (Blue) - points along +Z (no rotation, already points along -Z)
        self.z_arrow = ArrowObject(
            f"{name}_Z",
            stick_width=stick_width,
            stick_length=stick_length,
            triangle_width=triangle_width,
            triangle_length=triangle_length,
            arrow_color=(0.0, 0.0, 1.0)  # Blue
        )
        # No rotation needed - arrow naturally points along -Z
        self.add_child(self.z_arrow)
        
        # Store original arrows for remapping
        self._original_arrows = [self.x_arrow, self.y_arrow, self.z_arrow]
        self._original_colors = [
            np.array([1.0, 0.0, 0.0, 1.0]),  # Red
            np.array([0.0, 1.0, 0.0, 1.0]),  # Green
            np.array([0.0, 0.0, 1.0, 1.0])   # Blue
        ]
    
    def set_all_arrow_parameters(self, stick_width: float = None, stick_length: float = None,
                                 triangle_width: float = None, triangle_length: float = None):
        """
        Update arrow geometry parameters for all three arrows.
        
        Args:
            stick_width: Width of the arrow shaft (None to keep current)
            stick_length: Length of the arrow shaft (None to keep current)
            triangle_width: Width of the arrow head base (None to keep current)
            triangle_length: Length of the arrow head (None to keep current)
        """
        for arrow in [self.x_arrow, self.y_arrow, self.z_arrow]:
            arrow.set_arrow_parameters(stick_width, stick_length, triangle_width, triangle_length)
    
    def set_x_visible(self, visible: bool):
        """Show or hide the X axis arrow."""
        self.x_arrow.set_visible(visible)
    
    def set_y_visible(self, visible: bool):
        """Show or hide the Y axis arrow."""
        self.y_arrow.set_visible(visible)
    
    def set_z_visible(self, visible: bool):
        """Show or hide the Z axis arrow."""
        self.z_arrow.set_visible(visible)
    
    def set_visible(self, visible: bool):
        """Show or hide all three arrows."""
        self.x_arrow.set_visible(visible)
        self.y_arrow.set_visible(visible)
        self.z_arrow.set_visible(visible)
    
    def set_alpha(self, alpha: float):
        """
        Set the alpha/transparency for all three arrows.
        
        Args:
            alpha: Alpha value from 0.0 (fully transparent) to 1.0 (fully opaque)
        """
        self.x_arrow.set_arrow_alpha(alpha)
        self.y_arrow.set_arrow_alpha(alpha)
        self.z_arrow.set_arrow_alpha(alpha)
    
    def set_axes_visible(self, x: bool, y: bool, z: bool):
        """
        Set visibility for each axis individually.
        
        Args:
            x: Show/hide X axis
            y: Show/hide Y axis
            z: Show/hide Z axis
        """
        self.x_arrow.set_visible(x)
        self.y_arrow.set_visible(y)
        self.z_arrow.set_visible(z)
    
    def set_axis_order(self, order: str|AxisOrder):
        """
        Set the axis order for remapping.
        
        Args:
            order: List of 3 integers [x, y, z] where order[i] = j means
                   original axis i should go to display position j.
                   E.g., [2, 0, 1] means X->Z, Y->X, Z->Y
        """
        if isinstance(order, str):
            order = AxisOrder(order)
        if len(order.order) != 3 or set(order.order) != {0, 1, 2}:
            raise ValueError("order must be a permutation of x, y, and z (e.g., 'xyz', 'zyx', 'xzy', etc.)")
        self.axis_order = order
        self._update_axis_mapping()
    
    def _update_axis_mapping(self):
        """
        Update arrow positions and orientations based on axis_order and axis_multipliers.
        
        axis_order[i] = j means original axis i should go to new position j
        - Original axis 0 (X/Red) 
        - Original axis 1 (Y/Green)
        - Original axis 2 (Z/Blue)
        
        New positions:
        - Position 0 = X display location
        - Position 1 = Y display location
        - Position 2 = Z display location
        
        The multipliers rotate arrow 180° when negative.
        
        Note: self.x_arrow, self.y_arrow, self.z_arrow always refer to the semantic
        red, green, blue arrows respectively, regardless of where they're positioned.
        """
        # Standard rotations for each display axis position
        # X axis: -90° around Y
        # Y axis: +90° around X
        # Z axis: no rotation (arrow naturally points along -Z)
        base_rotations = [
            self._get_rotation_matrix(-90, [0, 1, 0]),  # X position
            self._get_rotation_matrix(90, [1, 0, 0]),   # Y position
            np.eye(3)                                    # Z position
        ]
        
        # 180° rotation around Y axis (for flipping)
        flip_rotation = self._get_rotation_matrix(180, [0, 1, 0])
        
        # For each original arrow, determine where it should go and configure it
        for orig_axis_idx in range(3):
            # Find which new position this original axis should go to
            new_axis_idx = self.axis_order.order[orig_axis_idx]
            
            # Get the original arrow object (these references never change)
            orig_arrow = self._original_arrows[orig_axis_idx]
            
            # Apply the base rotation for the target display position
            rotation = base_rotations[new_axis_idx].copy()
            
            # If multiplier is negative, add 180° rotation
            multiplier = self.axis_order.multipliers[orig_axis_idx]
            if multiplier < 0:
                rotation = rotation @ flip_rotation
            
            orig_arrow.set_rotation_matrix(rotation)
    
    def _get_rotation_matrix(self, angle_deg: float, axis: list) -> np.ndarray:
        """Helper to create rotation matrix using Rodrigues' formula."""
        if angle_deg == 0:
            return np.eye(3)
        
        angle_rad = np.radians(angle_deg)
        axis = np.array(axis, dtype=np.float32)
        axis = axis / (np.linalg.norm(axis) + 1e-6)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)
        return R
    
    def __getitem__(self, index: int) -> ArrowObject:
        """Get arrow by index: 0=x_arrow, 1=y_arrow, 2=z_arrow."""
        if index == 0:
            return self.x_arrow
        elif index == 1:
            return self.y_arrow
        elif index == 2:
            return self.z_arrow
        else:
            raise IndexError(f"TriAxesObject index out of range: {index} (valid range: 0-2)")
    
    def render(self, shader_program: int = None, parent_matrix: np.ndarray = None, *, scene_context: SceneContext = None) -> None:
        """TriAxes objects don't render themselves, only their arrow children."""
        pass


class LabeledTriAxesObject(TriAxesObject):
    """
    TriAxes with text labels (X, Y, Z) positioned at the end of each arrow.
    Labels are colored to match their respective axes: Red (X), Green (Y), Blue (Z).
    
    Example usage:
        # Create labeled axes
        axes = LabeledTriAxesObject("LabeledAxes", 
                                    font=my_font,
                                    stick_length=0.75,
                                    text_size=1.0,
                                    text_offset=3.0)
        
        # Add to scene
        scene.add_child(axes)
        
        # Access individual labels
        axes.x_label.set_text("Forward")
        axes.y_label.set_color(1.0, 1.0, 0.0)  # Yellow
    """
    
    def __init__(self, name: str = "LabeledTriAxes",
                 font: Font = None,
                 stick_width: float = 0.03,
                 stick_length: float = 0.75,
                 triangle_width: float = 0.15,
                 triangle_length: float = 0.25,
                 text_size: float = 0.25,
                 text_offset: float = 0.2):
        """
        Initialize labeled tri-axes with text labels.
        
        Args:
            name: Object name
            font: Font object to use for text labels
            stick_width: Width of the arrow shaft for all arrows
            stick_length: Length of the arrow shaft for all arrows
            triangle_width: Width of the arrow head base for all arrows
            triangle_length: Length of the arrow head for all arrows
            text_size: Size of text in world units (default 1.0)
            text_offset: Distance from origin to position the text labels
        """
        # Initialize parent TriAxesObject
        super().__init__(name, stick_width, stick_length, triangle_width, triangle_length)
        
        self.font = font
        self.text_size = text_size
        self.text_offset = text_offset
        
        # Calculate total arrow length for positioning labels
        total_arrow_length = stick_length + triangle_length
        
        # Create X label (Red) - positioned along +X axis
        self.x_label = TextMesh(
            text="X",
            font=font,
            name=f"{name}_X_Label",
            text_size=text_size,
            color=(1.0, 0.0, 0.0)  # Red
        )
        # Position at the end of the X arrow (plus offset)
        self.x_label.set_position([0.0, 0.0, -(total_arrow_length + text_offset)])
        self.x_arrow.add_child(self.x_label)
        
        # Create Y label (Green) - positioned along +Y axis
        self.y_label = TextMesh(
            text="Y",
            font=font,
            name=f"{name}_Y_Label",
            text_size=text_size,
            color=(0.0, 1.0, 0.0)  # Green
        )
        # Position at the end of the Y arrow (plus offset)
        self.y_label.set_position([0.0, 0.0, -(total_arrow_length + text_offset)])
        self.y_arrow.add_child(self.y_label)
        
        # Create Z label (Blue) - positioned along +Z axis
        self.z_label = TextMesh(
            text="Z",
            font=font,
            name=f"{name}_Z_Label",
            text_size=text_size,
            color=(0.0, 0.0, 1.0)  # Blue
        )
        # Position at the end of the Z arrow (plus offset)
        # Note: arrows point along -Z, so we use negative offset
        self.z_label.set_position([0.0, 0.0, -(total_arrow_length + text_offset)])
        self.z_arrow.add_child(self.z_label)
        
        # Store original labels for remapping
        self._original_labels = [self.x_label, self.y_label, self.z_label]
        self._original_label_texts = ["X", "Y", "Z"]
    
    def set_font(self, font: Font) -> None:
        """Set the font for all text labels."""
        self.font = font
        self.x_label.set_font(font)
        self.y_label.set_font(font)
        self.z_label.set_font(font)
    
    def set_text_size(self, text_size: float) -> None:
        """Set the size for all text labels in world units."""
        self.text_size = text_size
        self.x_label.set_text_size(text_size)
        self.y_label.set_text_size(text_size)
        self.z_label.set_text_size(text_size)
    
    def set_text_offset(self, offset: float) -> None:
        """
        Set the offset distance for all text labels from the arrow tips.
        
        Args:
            offset: Distance from arrow tip to label position
        """
        self.text_offset = offset
        total_arrow_length = (self.x_arrow.stick_length + self.x_arrow.triangle_length)
        
        # Update label positions
        self.x_label.set_position([0.0, 0.0, -(total_arrow_length + offset)])
        self.y_label.set_position([0.0, 0.0, -(total_arrow_length + offset)])
        self.z_label.set_position([0.0, 0.0, -(total_arrow_length + offset)])
    
    def set_all_arrow_parameters(self, stick_width: float = None, stick_length: float = None,
                                 triangle_width: float = None, triangle_length: float = None):
        """
        Update arrow geometry parameters for all three arrows.
        Also updates label positions to match new arrow lengths.
        
        Args:
            stick_width: Width of the arrow shaft (None to keep current)
            stick_length: Length of the arrow shaft (None to keep current)
            triangle_width: Width of the arrow head base (None to keep current)
            triangle_length: Length of the arrow head (None to keep current)
        """
        # Update arrows via parent class
        super().set_all_arrow_parameters(stick_width, stick_length, triangle_width, triangle_length)
        
        # Update label positions if length changed
        if stick_length is not None or triangle_length is not None:
            total_arrow_length = (self.x_arrow.stick_length + self.x_arrow.triangle_length)
            self.x_label.set_position([total_arrow_length + self.text_offset, 0.0, 0.0])
            self.y_label.set_position([0.0, total_arrow_length + self.text_offset, 0.0])
            self.z_label.set_position([0.0, 0.0, -(total_arrow_length + self.text_offset)])
    
    def set_x_label_visible(self, visible: bool):
        """Show or hide the X label."""
        self.x_label.set_visible(visible)
    
    def set_y_label_visible(self, visible: bool):
        """Show or hide the Y label."""
        self.y_label.set_visible(visible)
    
    def set_z_label_visible(self, visible: bool):
        """Show or hide the Z label."""
        self.z_label.set_visible(visible)
    
    def set_all_labels_visible(self, visible: bool):
        """Show or hide all labels."""
        self.x_label.set_visible(visible)
        self.y_label.set_visible(visible)
        self.z_label.set_visible(visible)
    
    def __getitem__(self, index: int) -> tuple[ArrowObject, TextMesh]:
        """Get (arrow, label) tuple by index: 0=(x_arrow, x_label), 1=(y_arrow, y_label), 2=(z_arrow, z_label)."""
        if index == 0:
            return (self.x_arrow, self.x_label)
        elif index == 1:
            return (self.y_arrow, self.y_label)
        elif index == 2:
            return (self.z_arrow, self.z_label)
        else:
            raise IndexError(f"LabeledTriAxesObject index out of range: {index} (valid range: 0-2)")
