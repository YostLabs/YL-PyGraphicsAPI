#version 330 core

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Color;
} fs_in;

out vec4 FragColor;

uniform sampler2D texture_diffuse;
uniform float modelAlpha;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec3 lightColor;

void main()
{
    // Use vertex color from material
    vec3 objectColor = fs_in.Color;
    
    // Normalize the normal
    vec3 norm = normalize(fs_in.Normal);
    
    // Ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * objectColor;

    // Diffuse
    vec3 lightDir = normalize(lightPos - fs_in.FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor * objectColor;

    // Specular
    float specularStrength = 0.2;
    vec3 viewDir = normalize(viewPos - fs_in.FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, modelAlpha);
}
