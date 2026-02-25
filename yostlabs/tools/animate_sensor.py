"""
Runnable module for visualizing Threespace Sensor orientation in real-time.

This module can be used in two ways:

1. Command line (auto-discover sensor):
   python -m yostlabs.tools.animate_sensor
   with optional arguments:
   --untared : Display untared orientation instead of tared
    --width WIDTH : Set window width in pixels (default: 600)
    --height HEIGHT : Set window height in pixels (default: 600)

2. Programmatically (provide your own sensor and options):
    from yostlabs.tools.animate_sensor import animate_sensor

    #Auto discover sensor and show tared orientation
    animate_sensor()

    OR

    from yostlabs.tss3.api import ThreespaceSensor
    sensor = ThreespaceSensor()
    animate_sensor(sensor=sensor, use_tared=True, window_width=800, window_height=600)

Requirements:
    pip install yostlabs-graphics[sensor]

Controls:
    - WASD + Space/Shift: Move camera
    - Mouse drag (right button): Rotate camera
    - ESC or close window: Exit
"""

import argparse
import sys

import glfw

from yostlabs.graphics import GL_Context, resources
from yostlabs.graphics import ModelObject
from yostlabs.graphics import scene_prefabs
from yostlabs.graphics.glfw import GlfwCameraMover
from yostlabs.math.axes import AxisOrder
from yostlabs.graphics import GL_AXIS_ORDER

# Import sensor library (optional dependency of yostlabs-graphics)
try:
    from yostlabs.tss3.api import ThreespaceSensor
    import yostlabs.tss3.consts as tss3_consts
    import serial
except ImportError:
    print("ERROR: yostlabs package not found.")
    print("Please install it with: pip install yostlabs")
    print("Or install with: pip install yostlabs-graphics[sensor]")
    sys.exit(1)

def animate_sensor(sensor: ThreespaceSensor = None, 
                   use_tared: bool = True,
                   window_width: int = 600,
                   window_height: int = 600) -> None:
    """
    Animate a Threespace Sensor's orientation in a 3D viewer window.
    
    Args:
        sensor: ThreespaceSensor object. If None, will auto-discover via USB.
        use_tared: If True, display tared orientation. If False, display untared orientation.
        window_width: Width of the window in pixels (default: 600)
        window_height: Height of the window in pixels (default: 600)
    
    Example:
        # Auto-discover sensor and show tared orientation
        animate_sensor()
        
        # Use existing sensor with untared orientation
        sensor = ThreespaceSensor()
        animate_sensor(sensor=sensor, use_tared=False)
    """
    
    # Initialize or validate sensor
    if sensor is None:
        try:
            print("Auto-discovering Threespace Sensor via USB...")
            sensor = ThreespaceSensor()
            print(f"Connected to sensor: {sensor.com.name}")
        except Exception as e:
            print(f"ERROR: Could not connect to sensor: {e}")
            print("Make sure a Threespace Sensor is connected via USB.")
            sys.exit(1)
    
    # Setup 3D Viewer
    orientation_mode = "Tared" if use_tared else "Untared"
    window_title = f"Threespace Sensor - {orientation_mode} Orientation"
    
    GL_Context.init(window_width=window_width, window_height=window_height, 
                   window_title=window_title, visible=True)
    
    # Create model based on sensor type
    # Default to embedded model, but could be extended to detect sensor type
    if sensor.sensor_family == tss3_consts.THREESPACE_FAMILY_EMBEDDED:
        model = ModelObject("EM", resources.get_model_path('EM-3.obj', 'Embedded'))
    elif sensor.sensor_family == tss3_consts.THREESPACE_FAMILY_DATA_LOGGER:
        model = ModelObject("DL", resources.get_model_path('DL-3.obj', 'DataLogger'))
    else:
        model = ModelObject("EM", resources.get_model_path('EM-3.obj', 'Embedded'))
    
    # Create OrientationScene with model and axes
    orientation_scene = scene_prefabs.OrientationScene(
        window_width,
        window_height,
        model,
        font=GL_Context.default_font,
        name="Sensor Scene"
    )
    orientation_scene.set_background_color(105 / 255, 105 / 255, 105 / 255, 1.0)
    
    # Get the GLFW window
    window = GL_Context.get_window()
    
    # Set up GLFW camera mover for the orientation scene
    glfw_camera_mover = GlfwCameraMover(orientation_scene.camera, window, 
                                       camera_speed=30.0, rotation_speed=1.0)
    orientation_scene.set_camera_mover(glfw_camera_mover)
    
    # Get sensor axis order
    sensor_axis_order_str = sensor.get_settings("axis_order")
    SENSOR_AXIS_ORDER = AxisOrder(sensor_axis_order_str)
    orientation_scene.set_axis_order(SENSOR_AXIS_ORDER)
    print(f"Sensor axis order: {sensor_axis_order_str}")
    
    print(f"Displaying {orientation_mode} orientation")
    print("\nControls:")
    print("  WASD + Space/Shift: Move camera")
    print("  Mouse drag (right button): Rotate camera")
    print("  ESC or close window: Exit")
    
    # Main render loop
    try:
        while not glfw.window_should_close(window):
            # Handle GLFW events
            glfw.poll_events()
            
            # Check for ESC key to exit
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            
            # Update camera using the camera mover
            orientation_scene.update_camera_pos()
            orientation_scene.update_camera_rotation()
            
            # Get orientation from sensor
            if use_tared:
                orientation = sensor.getTaredOrientation().data
            else:
                orientation = sensor.getUntaredOrientation().data
            
            # Map sensor axes to OpenGL axes
            glQuat = SENSOR_AXIS_ORDER.swap_to(GL_AXIS_ORDER, orientation, rotational=True)
            model.set_rotation_quat(glQuat)
            
            # Render the scene directly to the window
            orientation_scene.render()
            
            # Swap buffers
            glfw.swap_buffers(window)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        print(f"\nClosing due to: {e}")
    except (serial.SerialTimeoutException, serial.SerialException):
        print("\nClosing due to connection error. Check sensor connection.")
    finally:
        # Cleanup
        GL_Context.cleanup()
        try:
            sensor.cleanup()
        except Exception:
            pass
        print("Viewer closed")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Animate Threespace Sensor orientation in a 3D viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m yostlabs.tools.animate_sensor
  python -m yostlabs.tools.animate_sensor --untared
  python -m yostlabs.tools.animate_sensor --width 800 --height 600
        """
    )
    
    parser.add_argument(
        '--untared', 
        action='store_true',
        help='Display untared orientation instead of tared (default: tared)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=600,
        help='Window width in pixels (default: 600)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=600,
        help='Window height in pixels (default: 600)'
    )
    
    args = parser.parse_args()
    
    # Run the animation
    animate_sensor(
        sensor=None,  # Auto-discover
        use_tared=not args.untared,
        window_width=args.width,
        window_height=args.height
    )


if __name__ == "__main__":
    main()
