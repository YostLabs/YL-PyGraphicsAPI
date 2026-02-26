import os
from OpenGL.GL import *
import numpy as np
import ctypes

class OBJ:
    """OBJ loader that uses VAO/VBO for rendering."""
    
    @classmethod
    def loadMaterial(cls, filename):
        contents = {}
        mtl = None
        dirname = os.path.dirname(filename)

        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'newmtl':
                mtl = contents[values[1]] = {}
            elif mtl is None:
                raise ValueError("mtl file doesn't start with newmtl stmt")
            elif values[0] == 'map_Kd':
                # Texture support for future use
                mtl[values[0]] = values[1]
                mtl['texture_Kd'] = None
            else:
                mtl[values[0]] = list(map(float, values[1:]))
        return contents

    def __init__(self, filename, swapyz=False, scale=1):
        """Loads a Wavefront OBJ file."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.generated = False
        self.scale = scale
        self.swapyz = swapyz
        
        # Modern rendering data
        self.vao = None
        self.vbo = None  # Single interleaved VBO
        self.ebo = None  # Element Buffer Object
        self.vertex_count = 0
        self.index_count = 0
        self.mtl = {}
        
        # Local bounding box (based on vertices only)
        self.local_bbox_min = None
        self.local_bbox_max = None
        self.local_bbox_center = None
        self.local_bbox_size = None
        
        # Global bounding box (includes origin 0,0,0)
        self.global_bbox_min = None
        self.global_bbox_max = None
        self.global_bbox_center = None
        self.global_bbox_size = None
        
        dirname = os.path.dirname(filename)

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                v = v[0] * scale, v[1] * scale, v[2] * scale
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = ' '.join(values[1:])
            elif values[0] == 'mtllib':
                self.mtl = self.loadMaterial(os.path.join(dirname, ' '.join(values[1:])))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        
        # Calculate bounding box after loading all vertices
        self._calculate_bounding_box()
    
    def _calculate_bounding_box(self):
        """Calculate both local and global bounding boxes of the loaded model.
        
        Local bounding box: Based purely on the vertices of the model.
        Global bounding box: Includes the origin (0, 0, 0) as part of the bounding box.
        """
        if not self.vertices:
            self.local_bbox_min = np.array([0.0, 0.0, 0.0])
            self.local_bbox_max = np.array([0.0, 0.0, 0.0])
            self.local_bbox_center = np.array([0.0, 0.0, 0.0])
            self.local_bbox_size = np.array([0.0, 0.0, 0.0])
            self.global_bbox_min = np.array([0.0, 0.0, 0.0])
            self.global_bbox_max = np.array([0.0, 0.0, 0.0])
            self.global_bbox_center = np.array([0.0, 0.0, 0.0])
            self.global_bbox_size = np.array([0.0, 0.0, 0.0])
            return
        
        vertices_array = np.array(self.vertices)
        
        # Local bounding box (vertex-based only)
        self.local_bbox_min = vertices_array.min(axis=0)
        self.local_bbox_max = vertices_array.max(axis=0)
        self.local_bbox_center = (self.local_bbox_min + self.local_bbox_max) / 2.0
        self.local_bbox_size = self.local_bbox_max - self.local_bbox_min
        
        # Global bounding box (includes origin)
        origin = np.array([0.0, 0.0, 0.0])
        self.global_bbox_min = np.minimum(self.local_bbox_min, origin)
        self.global_bbox_max = np.maximum(self.local_bbox_max, origin)
        self.global_bbox_center = (self.global_bbox_min + self.global_bbox_max) / 2.0
        self.global_bbox_size = self.global_bbox_max - self.global_bbox_min
    
    def get_local_bounding_box(self):
        """Get the local bounding box (vertex-based only) as (min, max) tuple.
        
        Returns:
            tuple: (bbox_min, bbox_max) as numpy arrays with [x, y, z] coordinates
        """
        return (self.local_bbox_min, self.local_bbox_max)
    
    def get_local_bounding_box_center(self):
        """Get the center of the local bounding box.
        
        Returns:
            numpy.ndarray: Center position [x, y, z]
        """
        return self.local_bbox_center
    
    def get_local_bounding_box_size(self):
        """Get the size of the local bounding box.
        
        Returns:
            numpy.ndarray: Size dimensions [width, height, depth]
        """
        return self.local_bbox_size
    
    def get_local_max_dimension(self):
        """Get the maximum dimension of the local bounding box.
        
        Returns:
            float: Maximum of width, height, or depth
        """
        return np.max(self.local_bbox_size) if self.local_bbox_size is not None else 0.0
    
    def get_global_bounding_box(self):
        """Get the global bounding box (includes origin) as (min, max) tuple.
        
        Returns:
            tuple: (bbox_min, bbox_max) as numpy arrays with [x, y, z] coordinates
        """
        return (self.global_bbox_min, self.global_bbox_max)
    
    def get_global_bounding_box_center(self):
        """Get the center of the global bounding box.
        
        Returns:
            numpy.ndarray: Center position [x, y, z]
        """
        return self.global_bbox_center
    
    def get_global_bounding_box_size(self):
        """Get the size of the global bounding box.
        
        Returns:
            numpy.ndarray: Size dimensions [width, height, depth]
        """
        return self.global_bbox_size
    
    def get_global_max_dimension(self):
        """Get the maximum dimension of the global bounding box.
        
        Returns:
            float: Maximum of width, height, or depth
        """
        return np.max(self.global_bbox_size) if self.global_bbox_size is not None else 0.0

    def generate(self):
        """Generate VAO/VBO/EBO for modern indexed rendering"""
        if self.generated: return
        
        # Dictionary to track unique vertex combinations and their indices
        # Key: (vertex_idx, normal_idx, texcoord_idx, material)
        # Value: index in the unique vertex list
        unique_vertex_map = {}
        interleaved_data = []  # Format: pos(3) + normal(3) + texcoord(2) + color(3) = 11 floats per vertex
        indices = []
        
        for face in self.faces:
            vertices, normals, texture_coords, material = face
            
            # Get material color or use default
            material_color = [0.8, 0.8, 0.8]  # Default gray
            if material and material in self.mtl:
                mtl_data = self.mtl[material]
                if 'Kd' in mtl_data:
                    material_color = mtl_data['Kd'][:3]
            
            # Compute face normal once if needed for vertices without normals
            face_normal = None
            if len(vertices) >= 3 and any(n == 0 for n in normals):
                v0 = np.array(self.vertices[vertices[0] - 1])
                v1 = np.array(self.vertices[vertices[1] - 1])
                v2 = np.array(self.vertices[vertices[2] - 1])
                edge1 = v1 - v0
                edge2 = v2 - v0
                face_normal = np.cross(edge1, edge2)
                norm_len = np.linalg.norm(face_normal)
                if norm_len > 1e-6:
                    face_normal = face_normal / norm_len
                else:
                    face_normal = np.array([0, 1, 0])
            
            # Triangulate polygons with more than 3 vertices (fan triangulation)
            for tri_idx in range(1, len(vertices) - 1):
                # Triangle vertices: 0, tri_idx, tri_idx+1
                triangle_verts = [0, tri_idx, tri_idx + 1]
                
                for i in triangle_verts:
                    vert_idx = vertices[i]
                    norm_idx = normals[i]
                    tex_idx = texture_coords[i]
                    
                    # Create a unique key for this vertex combination
                    vertex_key = (vert_idx, norm_idx, tex_idx, material)
                    
                    # Check if we've seen this exact vertex combination before
                    if vertex_key in unique_vertex_map:
                        # Reuse existing vertex index
                        indices.append(unique_vertex_map[vertex_key])
                    else:
                        # Add new unique vertex with interleaved data
                        new_index = len(interleaved_data) // 11  # 11 floats per vertex
                        unique_vertex_map[vertex_key] = new_index
                        indices.append(new_index)
                        
                        # Add vertex position (3 floats)
                        if vert_idx > 0 and vert_idx - 1 < len(self.vertices):
                            interleaved_data.extend(self.vertices[vert_idx - 1])
                        else:
                            interleaved_data.extend([0, 0, 0])
                        
                        # Add normal (3 floats)
                        if norm_idx > 0 and norm_idx - 1 < len(self.normals):
                            interleaved_data.extend(self.normals[norm_idx - 1])
                        else:
                            # Use computed face normal or default
                            if face_normal is not None:
                                interleaved_data.extend(face_normal.tolist())
                            else:
                                interleaved_data.extend([0, 1, 0])
                        
                        # Add texture coordinates (2 floats)
                        if tex_idx > 0 and tex_idx - 1 < len(self.texcoords):
                            interleaved_data.extend(self.texcoords[tex_idx - 1])
                        else:
                            interleaved_data.extend([0, 0])
                        
                        # Add material color (3 floats)
                        interleaved_data.extend(material_color)
        
        self.vertex_count = len(interleaved_data) // 11  # 11 floats per vertex
        self.index_count = len(indices)
        
        # Create VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        
        # Single interleaved VBO with format: pos(3) + normal(3) + texcoord(2) + color(3)
        # Stride: 11 floats * 4 bytes = 44 bytes per vertex
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, np.array(interleaved_data, dtype=np.float32).nbytes,
                     np.array(interleaved_data, dtype=np.float32), GL_STATIC_DRAW)
        
        stride = 44  # 11 floats * 4 bytes
        
        # Position attribute (location 0): offset 0, 3 floats
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        
        # Normal attribute (location 1): offset 12 bytes (3 floats), 3 floats
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        
        # Texture coordinate attribute (location 2): offset 24 bytes (6 floats), 2 floats
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        
        # Color attribute (location 3): offset 32 bytes (8 floats), 3 floats
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(32))
        
        # Element Buffer Object (EBO) for indices
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, np.array(indices, dtype=np.uint32).nbytes,
                     np.array(indices, dtype=np.uint32), GL_STATIC_DRAW)
        
        glBindVertexArray(0)
        self.generated = True

    def render(self):
        """Render using VAO/EBO with indexed drawing"""
        if not self.generated:
            self.generate()
        
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def free(self):
        """Clean up GPU resources"""
        if self.vao is not None:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo is not None:
            glDeleteBuffers(1, [self.vbo])
        if self.ebo is not None:
            glDeleteBuffers(1, [self.ebo])
        self.vao = None
        self.vbo = None
        self.ebo = None
