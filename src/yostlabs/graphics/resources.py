"""
Resource utilities for accessing package data files (fonts, models, shaders).

Uses importlib.resources for proper package resource access that works
in all installation scenarios (development, installed, zip-safe, etc.).
"""

from importlib.resources import files
from pathlib import Path
from typing import Optional
import sys
import os


def get_package_path() -> Path:
    """
    Get the absolute path to the yostlabs.graphics package directory.
    
    Uses importlib.resources for proper package resource access.
    """
    package_files = files('yostlabs.graphics')
    # Convert to Path - this works even for zip-installed packages
    return Path(str(package_files))


def get_assets_root() -> Path:
    """
    Get the absolute path to the assets directory.
    
    For editable installs: looks for assets/ at the workspace root
    For regular installs: assets are copied to yostlabs/graphics/
    
    Returns:
        Path to the assets directory containing fonts/ and models/
    """
    package_path = get_package_path()
    
    # Check if assets exist in package directory (regular install)
    if (package_path / 'fonts').exists() or (package_path / 'models').exists():
        return package_path
    
    # For editable install, navigate up to workspace root and check for assets/
    # Package structure: .../workspace/yostlabs/graphics
    # Assets location: .../workspace/assets
    current = package_path
    
    # Try going up to find workspace root with assets folder
    for _ in range(5):  # Limit search depth
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        
        assets_dir = parent / 'assets'
        if assets_dir.exists() and assets_dir.is_dir():
            # Verify it has fonts or models subdirectories
            if (assets_dir / 'fonts').exists() or (assets_dir / 'models').exists():
                return assets_dir
        
        current = parent
    
    # Fallback to package path (will likely fail but provides clear error)
    return package_path


def get_font_path(font_name: str) -> Path:
    """
    Get the absolute path to a font file from package fonts or system fonts.
    
    Search order:
    1. Package fonts directory (exact match)
    2. Package fonts directory (with .ttf extension)
    3. System fonts directories
    
    Args:
        font_name: Name of the font file (e.g., 'arial.ttf', 'Arial.ttf')
                   or just the base name (e.g., 'arial', 'Arial')
    
    Returns:
        Path object pointing to the font file
    
    Raises:
        FileNotFoundError: If font cannot be found in package or system
    
    Example:
        # Load from package fonts
        font_path = get_font_path('FiraCode-Regular.ttf')
        
        # Load from system fonts
        font_path = get_font_path('arial.ttf')
        font_path = get_font_path('Arial')  # Will search for Arial.ttf
    """
    assets_root = get_assets_root()
    fonts_dir = assets_root / 'fonts'
    
    # 1. Try exact match in package fonts
    font_path = fonts_dir / font_name
    if font_path.exists():
        return font_path
    
    # 2. Try adding .ttf extension in package fonts
    if not font_name.lower().endswith('.ttf'):
        font_path = fonts_dir / f"{font_name}.ttf"
        if font_path.exists():
            return font_path
    
    # 3. Search system fonts
    system_font = _find_system_font(font_name)
    if system_font:
        return system_font
    
    # Not found - raise error with helpful message
    raise FileNotFoundError(
        f"Font '{font_name}' not found in package fonts or system fonts. "
        f"Searched in:\n"
        f"  - Package fonts: {fonts_dir}\n"
        f"  - System fonts: {', '.join(str(p) for p in _get_system_font_paths())}"
    )


def _get_system_font_paths() -> list[Path]:
    """Get platform-specific system font directories."""
    font_paths = []
    
    if sys.platform == 'win32':
        # Windows
        windows_fonts = Path(os.environ.get('WINDIR', 'C:\\Windows')) / 'Fonts'
        font_paths.append(windows_fonts)
        
        # User fonts
        local_app_data = os.environ.get('LOCALAPPDATA')
        if local_app_data:
            font_paths.append(Path(local_app_data) / 'Microsoft' / 'Windows' / 'Fonts')
    
    elif sys.platform == 'darwin':
        # macOS
        font_paths.extend([
            Path('/Library/Fonts'),
            Path('/System/Library/Fonts'),
            Path.home() / 'Library' / 'Fonts',
        ])
    
    else:
        # Linux/Unix
        font_paths.extend([
            Path('/usr/share/fonts'),
            Path('/usr/local/share/fonts'),
            Path.home() / '.fonts',
            Path.home() / '.local' / 'share' / 'fonts',
        ])
    
    # Return only existing directories
    return [p for p in font_paths if p.exists()]


def _find_system_font(font_name: str) -> Optional[Path]:
    """
    Find a font file in system font directories.
    
    Args:
        font_name: Font filename (e.g., 'arial.ttf' or 'Arial')
    
    Returns:
        Path to font file if found, None otherwise
    """
    # Normalize font name
    search_name = font_name.lower()
    if not search_name.endswith('.ttf'):
        search_name += '.ttf'
    
    # Search in all system font directories
    for font_dir in _get_system_font_paths():
        # Try direct match in root
        font_path = font_dir / search_name
        if font_path.exists():
            return font_path
        
        # Try case-insensitive match in root
        if font_dir.exists():
            for file in font_dir.iterdir():
                if file.is_file() and file.name.lower() == search_name:
                    return file
        
        # Search recursively (fonts can be in subdirectories)
        try:
            for file in font_dir.rglob('*.ttf'):
                if file.name.lower() == search_name:
                    return file
        except (PermissionError, OSError):
            # Skip directories we can't read
            continue
    
    return None


def list_available_fonts() -> list[str]:
    """
    Get a list of all available fonts from both package and system.
    
    Returns:
        Sorted list of font filenames (without duplicates)
    
    Example:
        fonts = list_available_fonts()
        print('Available fonts:', fonts[:10])  # First 10 fonts
    """
    all_fonts = set()
    
    # Get package fonts
    assets_root = get_assets_root()
    fonts_dir = assets_root / 'fonts'
    if fonts_dir.exists():
        for f in fonts_dir.iterdir():
            if f.is_file() and f.suffix.lower() in ['.ttf']:
                all_fonts.add(f.name)
    
    # Get system fonts
    for font_dir in _get_system_font_paths():
        try:
            for font_file in font_dir.rglob('*.ttf'):
                if font_file.is_file():
                    all_fonts.add(font_file.name)
        except (PermissionError, OSError):
            # Skip directories we can't read
            continue
    
    return sorted(list(all_fonts))


def get_model_path(model_name: str, subfolder: Optional[str] = None) -> Path:
    """
    Get the absolute path to a model file in the package.
    
    Args:
        model_name: Name of the model file (e.g., 'Camera.obj' or just 'Camera')
        subfolder: Optional subfolder within models/ (e.g., 'Camera', 'DataLogger')
    
    Returns:
        Path object pointing to the model file
    
    Example:
        # Load from models/Camera/Camera.obj
        model_path = get_model_path('Camera.obj', 'Camera')
        model = ModelObject('Camera', str(model_path))
        
        # Or let it auto-find
        model_path = get_model_path('Camera')  # Searches for Camera.obj
        model = ModelObject('Camera', str(model_path))
    """
    assets_root = get_assets_root()
    models_dir = assets_root / 'models'
    
    if subfolder:
        search_dir = models_dir / subfolder
    else:
        search_dir = models_dir
    
    # Try exact match first
    model_path = search_dir / model_name
    if model_path.exists():
        return model_path
    
    # Try adding .obj extension
    if not model_name.endswith('.obj'):
        model_path = search_dir / f"{model_name}.obj"
        if model_path.exists():
            return model_path
    
    # If no subfolder specified, try searching all subfolders
    if not subfolder:
        for item in models_dir.iterdir():
            if item.is_dir():
                # Try in this subfolder
                model_path = item / model_name
                if model_path.exists():
                    return model_path
                
                # Try with .obj extension
                if not model_name.endswith('.obj'):
                    model_path = item / f"{model_name}.obj"
                    if model_path.exists():
                        return model_path
    
    # If not found, return the path anyway (will fail when trying to load)
    return search_dir / model_name


def get_shader_path(shader_name: str) -> Path:
    """
    Get the absolute path to a shader file in the package.
    
    Args:
        shader_name: Name of the shader file (e.g., 'vertex.glsl')
    
    Returns:
        Path object pointing to the shader file
    
    Example:
        vertex_shader = get_shader_path('vertex.glsl')
    """
    package_path = get_package_path()
    shaders_dir = package_path / 'shaders'
    return shaders_dir / shader_name

def list_available_models() -> dict[str, list[str]]:
    """
    Get a dictionary of all available models organized by subfolder.
    
    Returns:
        Dictionary mapping subfolder names to lists of model files
    
    Example:
        {'Camera': ['Camera.obj'], 'DataLogger': ['DL3.obj'], ...}
    """
    assets_root = get_assets_root()
    models_dir = assets_root / 'models'
    
    if not models_dir.exists():
        return {}
    
    result = {}
    for item in models_dir.iterdir():
        if item.is_dir():
            models = [f.name for f in item.iterdir() if f.is_file() and f.suffix == '.obj']
            if models:
                result[item.name] = models
    
    return result
