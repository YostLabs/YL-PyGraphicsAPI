#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 96) out;

in VS_OUT {
    vec3 position;
} gs_in[];

out GS_OUT {
    vec3 FragPos;
    vec3 Normal;
    float Brightness; // For tip vs stick brightness variation
} gs_out;

uniform mat4 projection;
uniform mat4 view;
uniform mat4 model;
uniform float stickWidth;
uniform float stickLength;
uniform float triangleWidth;
uniform float triangleLength;

void emitQuad(vec3 v1, vec3 v2, vec3 v3, vec3 v4, vec3 normal, float brightness) {
    mat4 mvp = projection * view * model;
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    
    // Triangle 1
    gs_out.FragPos = vec3(model * vec4(v1, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v1, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v2, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v2, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v3, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v3, 1.0);
    EmitVertex();
    EndPrimitive();
    
    // Triangle 2
    gs_out.FragPos = vec3(model * vec4(v3, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v3, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v4, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v4, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v1, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v1, 1.0);
    EmitVertex();
    EndPrimitive();
}

void emitTriangle(vec3 v1, vec3 v2, vec3 v3, vec3 normal, float brightness) {
    mat4 mvp = projection * view * model;
    mat3 normalMatrix = mat3(transpose(inverse(model)));
    
    gs_out.FragPos = vec3(model * vec4(v1, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v1, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v2, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v2, 1.0);
    EmitVertex();
    
    gs_out.FragPos = vec3(model * vec4(v3, 1.0));
    gs_out.Normal = normalMatrix * normal;
    gs_out.Brightness = brightness;
    gl_Position = mvp * vec4(v3, 1.0);
    EmitVertex();
    EndPrimitive();
}

void main()
{
    float hsw = stickWidth / 2.0;
    float htw = triangleWidth / 2.0;
    float tipLength = stickLength + triangleLength;
    
    // Stick brightness
    float stickBrightness = 1.0;
    // Tip brightness (brighter)
    float tipBrightness = 1.3;
    
    // Back Face (CCW)
    emitQuad(
        vec3(-hsw, -hsw, 0.0), 
        vec3(hsw, -hsw, 0.0), 
        vec3(hsw, hsw, 0.0), 
        vec3(-hsw, hsw, 0.0), 
        vec3(0.0, 0.0, 1.0),
        stickBrightness
    );
    
    // Right (CCW)
    emitQuad(
        vec3(hsw, hsw, 0.0), 
        vec3(hsw, -hsw, 0.0), 
        vec3(hsw, -hsw, -stickLength), 
        vec3(hsw, hsw, -stickLength), 
        vec3(1.0, 0.0, 0.0),
        stickBrightness
    );
    
    // Top (CCW)
    emitQuad(
        vec3(-hsw, hsw, 0.0), 
        vec3(hsw, hsw, 0.0), 
        vec3(hsw, hsw, -stickLength), 
        vec3(-hsw, hsw, -stickLength), 
        vec3(0.0, 1.0, 0.0),
        stickBrightness
    );
    
    // Left (CCW)
    emitQuad(
        vec3(-hsw, hsw, 0.0), 
        vec3(-hsw, hsw, -stickLength), 
        vec3(-hsw, -hsw, -stickLength), 
        vec3(-hsw, -hsw, 0.0), 
        vec3(-1.0, 0.0, 0.0),
        stickBrightness
    );
    
    // Bottom (CCW)
    emitQuad(
        vec3(-hsw, -hsw, 0.0), 
        vec3(-hsw, -hsw, -stickLength), 
        vec3(hsw, -hsw, -stickLength), 
        vec3(hsw, -hsw, 0.0), 
        vec3(0.0, -1.0, 0.0),
        stickBrightness
    );
    
    // Pyramid Base (CCW)
    emitQuad(
        vec3(-htw, -htw, -stickLength), 
        vec3(htw, -htw, -stickLength), 
        vec3(htw, htw, -stickLength), 
        vec3(-htw, htw, -stickLength), 
        vec3(0.0, 0.0, 1.0),
        tipBrightness
    );
    
    vec3 tip = vec3(0.0, 0.0, -tipLength);
    
    // Right Triangle (CCW)
    vec3 br = vec3(htw, -htw, -stickLength);
    vec3 bl = vec3(htw, htw, -stickLength);
    vec3 base = br - bl;
    vec3 up = tip - bl;
    vec3 normal = normalize(cross(base, up));
    emitTriangle(br, tip, bl, normal, tipBrightness);
    
    // Top Triangle (CCW)
    br = vec3(htw, htw, -stickLength);
    bl = vec3(-htw, htw, -stickLength);
    base = br - bl;
    up = tip - bl;
    normal = normalize(cross(base, up));
    emitTriangle(br, tip, bl, normal, tipBrightness);
    
    // Left Triangle (CCW)
    br = vec3(-htw, htw, -stickLength);
    bl = vec3(-htw, -htw, -stickLength);
    base = br - bl;
    up = tip - bl;
    normal = normalize(cross(base, up));
    emitTriangle(br, tip, bl, normal, tipBrightness);
    
    // Bottom Triangle (CCW)
    br = vec3(-htw, -htw, -stickLength);
    bl = vec3(htw, -htw, -stickLength);
    base = br - bl;
    up = tip - bl;
    normal = normalize(cross(base, up));
    emitTriangle(br, tip, bl, normal, tipBrightness);
}
