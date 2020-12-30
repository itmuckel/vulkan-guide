#version 450 core

layout (location = 0) in vec3 inColor;

layout (location = 0) out vec4 outFragColor;

void main() 
{
	float hue = (inColor.r + inColor.g + inColor.b) / 3.f;
	outFragColor = vec4(vec3(0.f, hue, 0.f),1.0f);
}
