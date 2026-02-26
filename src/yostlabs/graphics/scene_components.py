from abc import ABC, abstractmethod
from yostlabs.graphics.core import Camera

class CameraMover(ABC):
    """
    Abstract base class for camera movement controllers.
    
    Defines the interface for camera movement and rotation control systems.
    Implementations should handle input from specific UI frameworks (DearPyGui, GLFW, etc.)
    and translate them into camera transformations.
    
    Example usage:
        # Create a camera mover (using a concrete implementation)
        mover = GlfwCameraMover(camera, window)
        
        # In main loop
        while running:
            mover.update_camera_pos()
            mover.update_camera_rotation()
    """
    
    def __init__(self, camera: Camera, camera_speed: float = 1.0, rotation_speed: float = 1.0):
        """
        Initialize the camera mover.
        
        Args:
            camera: The Camera object to control
            camera_speed: Camera movement speed in units per second
            rotation_speed: Camera rotation sensitivity. Higher values = faster rotation.
        """
        self.camera = camera
        self._camera_speed = camera_speed
        self._rotation_speed = rotation_speed
    
    @abstractmethod
    def update_camera_pos(self) -> None:
        """
        Update camera position based on input.
        Call this in your main loop if you want keyboard camera controls.
        """
        pass
    
    @abstractmethod
    def update_camera_rotation(self) -> None:
        """
        Update camera rotation based on input.
        Call this in your main loop if you want mouse camera controls.
        """
        pass
    
    def set_camera_speed(self, speed: float) -> None:
        """Set the camera movement speed (units per second)."""
        self._camera_speed = speed
    
    def reset_virtual_mouse(self) -> None:
        """Reset any virtual mouse tracking. Override if needed."""
        pass