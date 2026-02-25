import glfw

from yostlabs.graphics.core import Camera
from yostlabs.graphics.scene_components import CameraMover

from yostlabs.math.quaternion import quat_mul, quat_from_axis_angle

import numpy as np
import math
import time

class GlfwCameraMover(CameraMover):
    """
    Handles camera movement and rotation using GLFW input.
    
    Provides WASD + Space/Shift keyboard controls for camera position
    and right-click drag mouse controls for camera rotation.
    
    Example usage:
        # Create camera mover
        window = glfw.create_window(...)
        mover = GlfwCameraMover(camera, window)
        
        # In main loop
        while running:
            mover.update_camera_pos()
            mover.update_camera_rotation()
    """
    
    def __init__(self, camera: Camera, window, camera_speed: float = 1.0, rotation_speed: float = 1.0):
        """
        Initialize the GLFW camera mover.
        
        Args:
            camera: The Camera object to control
            window: GLFW window handle
            camera_speed: Camera movement speed in units per second
            rotation_speed: Camera rotation sensitivity. Higher values = faster rotation.
        """
        super().__init__(camera, camera_speed, rotation_speed)
        self.window = window
        
        # Camera position control state
        self._last_update_time = None
        
        # Camera rotation control state
        self._last_mouse_position = None
        self._virtual_mouse_position = np.array([0, 0], dtype=np.float32)
        self._was_right_mouse_down = False
    
    def update_camera_pos(self) -> None:
        """
        Update camera position based on keyboard input (WASD + Space/Shift).
        Call this in your main loop if you want keyboard camera controls.
        """
        forward = 0
        right = 0
        up = 0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            forward += 1
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            forward -= 1
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            right -= 1
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            right += 1
        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            up += 1
        if glfw.get_key(self.window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            up -= 1

        cur_time = time.time()
        if self._last_update_time is None:
            self._last_update_time = cur_time
        delta_time = cur_time - self._last_update_time

        self.camera.position += self.camera.get_forward() * forward * self._camera_speed * delta_time
        self.camera.position += self.camera.get_right() * right * self._camera_speed * delta_time
        self.camera.position += self.camera.get_up() * up * self._camera_speed * delta_time

        self._last_update_time = cur_time
    
    def update_camera_rotation(self) -> None:
        """
        Update camera rotation based on mouse input (right-click drag).
        Call this in your main loop if you want mouse camera controls.
        """
        is_right_mouse_down = glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        if is_right_mouse_down:
            mouse_x, mouse_y = glfw.get_cursor_pos(self.window)
            real_mouse_pos = np.array([mouse_x, mouse_y], dtype=np.float32)
            # Right mouse is currently down
            if not self._was_right_mouse_down:
                # Just pressed - compute offset to maintain continuity and prevent camera jump
                self._last_mouse_position = real_mouse_pos
            
            # Compute current virtual mouse offset while dragging
            delta = real_mouse_pos - self._last_mouse_position
            self._virtual_mouse_position += delta

            # Update last position
            self._last_mouse_position = real_mouse_pos
        
        # Update state for next frame
        self._was_right_mouse_down = is_right_mouse_down

        # Compute scale factor for mouse-to-rotation conversion
        # Base scale of 600, inversely proportional to rotation speed
        rotation_scale = 600.0 / self._rotation_speed

        # Clamp pitch to prevent exceeding ±89°
        max_pitch_offset = 89 * rotation_scale / 180
        if abs(self._virtual_mouse_position[1]) > max_pitch_offset:
            self._virtual_mouse_position[1] = np.sign(self._virtual_mouse_position[1]) * max_pitch_offset

        yaw = math.radians(-self._virtual_mouse_position[0] / rotation_scale * 360)
        pitch = math.radians(-self._virtual_mouse_position[1] / rotation_scale * 180)
        pitch = max(-math.radians(89), min(math.radians(89), pitch))

        rotation = quat_mul(quat_from_axis_angle([0,1,0], yaw), quat_from_axis_angle([1,0,0], pitch))
        self.camera.set_rotation_quat(rotation)
    
    def reset_virtual_mouse(self) -> None:
        """Reset virtual mouse position tracking."""
        self._virtual_mouse_position = np.array([0, 0], dtype=np.float32)
        self._last_mouse_position = None
        self._was_right_mouse_down = False