#include "vk_mesh.h"
#include <tiny_obj_loader.h>
#include <iostream>

VertexInputDescription Vertex::getVertexDescription()
{
	VertexInputDescription description;

	// we will have just 1 vertex buffer binding, with a per-vertex rate
	VkVertexInputBindingDescription mainBinding{};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	VkVertexInputAttributeDescription positionAttribute{};
	positionAttribute.binding = 0;
	positionAttribute.location = 0;
	positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	positionAttribute.offset = offsetof(Vertex, position);

	VkVertexInputAttributeDescription normalAttribute{};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	normalAttribute.offset = offsetof(Vertex, normal);

	VkVertexInputAttributeDescription colorAttribute{};
	colorAttribute.binding = 0;
	colorAttribute.location = 2;
	colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
	colorAttribute.offset = offsetof(Vertex, color);

	VkVertexInputAttributeDescription uvAttribute{};
	uvAttribute.binding = 0;
	uvAttribute.location = 3;
	uvAttribute.format = VK_FORMAT_R32G32_SFLOAT;
	uvAttribute.offset = offsetof(Vertex, uv);

	description.attributes.push_back(positionAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(colorAttribute);
	description.attributes.push_back(uvAttribute);

	return description;
}

bool Mesh::loadFromObj(const char* filename)
{
	tinyobj::attrib_t attrib{};
	std::vector<tinyobj::shape_t> shapes{};
	std::vector<tinyobj::material_t> materials{};

	std::string warn{};
	std::string err{};

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

	if (!warn.empty())
	{
		std::cout << "WARN: " << warn << '\n';
	}

	if (!err.empty())
	{
		std::cerr << err << '\n';
		return false;
	}

	// loop over shapes
	for (const auto& shape : shapes)
	{
		// loop over faces(polygon)
		size_t indexOffset = 0;
		for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
		{
			// hardcode loading to triangles
			int fv = 3;

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++)
			{
				// access to vertex
				const auto idx = shape.mesh.indices[indexOffset + v];

				// vertex position
				const auto vx = attrib.vertices[3 * idx.vertex_index + 0];
				const auto vy = attrib.vertices[3 * idx.vertex_index + 1];
				const auto vz = attrib.vertices[3 * idx.vertex_index + 2];
				// vertex normal
				const auto nx = attrib.normals[3 * idx.normal_index + 0];
				const auto ny = attrib.normals[3 * idx.normal_index + 1];
				const auto nz = attrib.normals[3 * idx.normal_index + 2];
				// vertex uv
				const auto ux = attrib.texcoords[2 * idx.texcoord_index + 0];
				const auto uy = attrib.texcoords[2 * idx.texcoord_index + 1];

				// copy it into our vertex
				Vertex newVert{};
				newVert.position.x = vx;
				newVert.position.y = vy;
				newVert.position.z = vz;

				newVert.normal.x = nx;
				newVert.normal.y = ny;
				newVert.normal.z = nz;

				newVert.uv.x = ux;
				newVert.uv.y = 1 - uy;

				// we are setting the vertex color as the vertex normal. This is just for display purposes
				newVert.color = newVert.normal;


				vertices.push_back(newVert);
			}

			indexOffset += fv;
		}
	}

	return true;
}
