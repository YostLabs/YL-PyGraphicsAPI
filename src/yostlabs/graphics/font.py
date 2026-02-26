from OpenGL.GL import *
from OpenGL.GL import shaders
import freetype
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Glyph:
    texture_id: int         #Open GL texture handle
    size: np.ndarray        #Size of glyph
    bearing: np.ndarray     #Offset from baseline to left/top of glyph
    advance: int            #Offset to next glyph (In 1/64th pixels)

class Font:

    def __init__(self, path: str|Path, resolution: int = 64):
        """
        Initialize a font.
        
        Args:
            path: Path to the font file
            resolution: Height in pixels to rasterize glyphs (affects texture quality).
                       Higher values = sharper text. Default 64 is good for most cases.
                       Use 128+ for very large text that will be viewed up close.
        """
        self.face = freetype.Face(str(path))
        self.face.set_pixel_sizes(0, resolution)
        self.resolution = resolution

        self.glyphs: dict[str, Glyph] = {}

    def load_character(self, char: str):
        if char in self.glyphs: return

        self.face.load_char(char)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, self.face.glyph.bitmap.width, self.face.glyph.bitmap.rows, 0, GL_RED, GL_UNSIGNED_BYTE, self.face.glyph.bitmap.buffer)

        #Set texture options
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        #Save off the texture
        glyph = Glyph(texture,
                      np.array([self.face.glyph.bitmap.width, self.face.glyph.bitmap.rows], dtype=np.int32),   #Size
                      np.array([self.face.glyph.bitmap_left, self.face.glyph.bitmap_top], dtype=np.int32),     #Bearing
                      self.face.glyph.advance.x
                      )
        self.glyphs[char] = glyph

        glPixelStorei(GL_UNPACK_ALIGNMENT, 4) #Set back

class TextRenderer:

    VERTEX_SOURCE = \
"""
#version 330 core
layout (location = 0) in vec4 vertex; // <vec2 pos, vec2 tex>
out vec2 TexCoords;

uniform mat4 projection;

void main()
{
    gl_Position = projection * vec4(vertex.xy, 0.0, 1.0);
    TexCoords = vertex.zw;
}  
"""

    FRAG_SOURCE = \
"""
#version 330 core
in vec2 TexCoords;
out vec4 color;

uniform sampler2D text;
uniform vec3 textColor;

void main()
{    
    vec4 sampled = vec4(1.0, 1.0, 1.0, texture(text, TexCoords).r);
    color = vec4(textColor, 1.0) * sampled;
    
    // Discard fully transparent pixels so they don't write to depth buffer
    if (color.a < 0.01)
        discard;
}
"""

    def __init__(self):
        self.textProgram = shaders.compileProgram(
            shaders.compileShader(TextRenderer.VERTEX_SOURCE, GL_VERTEX_SHADER),
            shaders.compileShader(TextRenderer.FRAG_SOURCE, GL_FRAGMENT_SHADER)
        )

        # configure VAO/VBO for texture quads
        # -----------------------------------
        self.VAO = glGenVertexArrays(1)
        self.VBO = glGenBuffers(1)
        glBindVertexArray(self.VAO)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, sizeof(ctypes.c_float) * 6 * 4, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(ctypes.c_float), None)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        
        # Store projection and view matrices
        self.projection_matrix = np.identity(4, dtype=np.float32)
        self.view_matrix = np.identity(4, dtype=np.float32)
    
    def set_projection_matrix(self, projection: np.ndarray) -> None:
        """Set the projection matrix for text rendering."""
        self.projection_matrix = np.array(projection, dtype=np.float32)
    
    def set_view_matrix(self, view: np.ndarray) -> None:
        """Set the view matrix for text rendering."""
        self.view_matrix = np.array(view, dtype=np.float32)

    def render_text(self, font: Font, text: str, x: float, y: float, text_size: float, color: list, model_matrix: np.ndarray = None, centered = False):
        """
        Render text at a specific world size.
        
        Args:
            font: Font to use for rendering
            text: Text string to render
            x: X position (relative to model_matrix)
            y: Y position (relative to model_matrix)
            text_size: Height of text in world units (e.g., 1.0 = 1 unit tall)
            color: RGB color tuple
            model_matrix: Transform matrix for positioning
            centered: Whether to center text at position
        """
        if len(text) == 0: return
        glUseProgram(self.textProgram)
        glUniform3f(glGetUniformLocation(self.textProgram, "textColor"), *color[:3])
        
        # Compute full projection * view * model matrix
        if model_matrix is None:
            model_matrix = np.identity(4, dtype=np.float32)
        mv = self.view_matrix @ model_matrix
        mv[:3,:3] = np.identity(3) # No rotation for text
        full_matrix = self.projection_matrix @ mv
        
        glUniformMatrix4fv(glGetUniformLocation(self.textProgram, "projection"), 1, GL_TRUE, full_matrix)
        glActiveTexture(GL_TEXTURE0)
        glBindVertexArray(self.VAO)

        # Blending is already enabled globally, just set blend function
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # Depth writing stays enabled - shader discards transparent pixels
        
        # Calculate scale to convert font resolution to desired world size
        # text_size is in world units, font.resolution is in pixels
        effective_scale = text_size / font.resolution

        #First character starts centered at given position
        if centered:
            c = text[0]
            if c not in font.glyphs:
                font.load_character(c)
            glyph = font.glyphs[c]

            w = glyph.size[0] * effective_scale
            h = glyph.size[1] * effective_scale

            x -= glyph.bearing[0] * effective_scale
            x -= w / 2

            y -= (glyph.size[1] - glyph.bearing[1]) * effective_scale
            y -= h / 2

        for c in text:
            if c not in font.glyphs:
                font.load_character(c)
            glyph = font.glyphs[c]


            xpos = x + glyph.bearing[0] * effective_scale
            ypos = y - (glyph.size[1] - glyph.bearing[1]) * effective_scale

            w = glyph.size[0] * effective_scale
            h = glyph.size[1] * effective_scale

            #Update VBO
            vertices = np.array([
                [ xpos,     ypos + h,   0.0, 0.0 ],            
                [ xpos,     ypos,       0.0, 1.0 ],
                [ xpos + w, ypos,       1.0, 1.0 ],

                [ xpos,     ypos + h,   0.0, 0.0 ],
                [ xpos + w, ypos,       1.0, 1.0 ],
                [ xpos + w, ypos + h,   1.0, 0.0 ]
            ], dtype=np.float32)
          
            glBindTexture(GL_TEXTURE_2D, glyph.texture_id)

            #Update the VBO so the shape of the quad matches the shape of the character
            glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
            glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

            glDrawArrays(GL_TRIANGLES, 0, 6)

            x += (glyph.advance >> 6) * effective_scale  
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)
