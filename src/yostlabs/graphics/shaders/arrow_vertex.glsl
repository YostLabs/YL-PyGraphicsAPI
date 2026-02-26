#version 330 core

layout (location = 0) in vec3 position;

out VS_OUT {
    vec3 position;
} vs_out;

void main()
{
    vs_out.position = position;
    gl_Position = vec4(position, 1.0);
}
