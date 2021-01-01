// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>

namespace vkinit
{
	VkCommandPoolCreateInfo commandPoolCreateInfo(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags = 0);

	VkCommandBufferAllocateInfo commandBufferAllocateInfo(VkCommandPool pool, uint32_t count = 1,
	                                                      VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo(VkShaderStageFlagBits stage,
	                                                              VkShaderModule shaderModule);

	VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo();

	VkPipelineInputAssemblyStateCreateInfo inputAssemblyCreateInfo(VkPrimitiveTopology topology);

	VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo(VkPolygonMode polygonMode);

	VkPipelineMultisampleStateCreateInfo multisamplingStateCreateInfo();

	VkPipelineColorBlendAttachmentState colorBlendAttachmentState();

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo();

	VkRenderPassBeginInfo renderPassBeginInfo(const VkRenderPass& renderPass, const VkExtent2D& windowExtent,
	                                          const VkFramebuffer& framebuffer);

	VkImageCreateInfo imageCreateInfo(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);

	VkImageViewCreateInfo imageviewCreateInfo(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

	VkPipelineDepthStencilStateCreateInfo depthStencilCreateInfo(
		bool depthTest, bool depthWrite, VkCompareOp compareOp);

	VkDescriptorSetLayoutBinding descriptorsetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stageFlags,
	                                                        uint32_t binding);

	VkWriteDescriptorSet writeDescriptorBuffer(VkDescriptorType type, VkDescriptorSet dstSet,
	                                           VkDescriptorBufferInfo* bufferInfo, uint32_t binding);
}
