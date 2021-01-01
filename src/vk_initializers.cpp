#include <vk_initializers.h>

// ----------------------------- commands

VkCommandPoolCreateInfo vkinit::commandPoolCreateInfo(uint32_t queueFamilyIndex, const VkCommandPoolCreateFlags flags)
{
	VkCommandPoolCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	info.queueFamilyIndex = queueFamilyIndex;
	info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

	return info;
}

VkCommandBufferAllocateInfo vkinit::commandBufferAllocateInfo(const VkCommandPool pool, uint32_t count,
                                                              const VkCommandBufferLevel level)
{
	VkCommandBufferAllocateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	info.commandPool = pool;
	info.commandBufferCount = 1;
	info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

	return info;
}

// ----------------------------- pipelines

VkPipelineShaderStageCreateInfo vkinit::pipelineShaderStageCreateInfo(const VkShaderStageFlagBits stage,
                                                                      const VkShaderModule shaderModule)
{
	VkPipelineShaderStageCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

	info.stage = stage;
	// module containing the code for this shader stage
	info.module = shaderModule;
	// the entry point of the shader
	info.pName = "main";

	return info;
}

/// This is the equivalent to OpenGL's VAO
VkPipelineVertexInputStateCreateInfo vkinit::vertexInputStateCreateInfo()
{
	VkPipelineVertexInputStateCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

	//no vertex bindings or attributes
	info.vertexBindingDescriptionCount = 0;
	info.vertexAttributeDescriptionCount = 0;
	return info;
}

VkPipelineInputAssemblyStateCreateInfo vkinit::inputAssemblyCreateInfo(const VkPrimitiveTopology topology)
{
	VkPipelineInputAssemblyStateCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;

	info.topology = topology;
	info.primitiveRestartEnable = VK_FALSE;

	return info;
}

VkPipelineRasterizationStateCreateInfo vkinit::rasterizationStateCreateInfo(const VkPolygonMode polygonMode)
{
	VkPipelineRasterizationStateCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;

	info.depthClampEnable = VK_FALSE;
	// discards all primitives before the rasterization stage if enabled which we don't want
	info.rasterizerDiscardEnable = VK_FALSE;

	info.polygonMode = polygonMode;
	info.lineWidth = 1.0f;
	// no backface cull
	info.cullMode = VK_CULL_MODE_NONE;
	info.frontFace = VK_FRONT_FACE_CLOCKWISE;
	// no depth bias
	info.depthBiasEnable = VK_FALSE;
	info.depthBiasConstantFactor = 0.0f;
	info.depthBiasClamp = 0.0f;
	info.depthBiasSlopeFactor = 0.0f;

	return info;
}

VkPipelineMultisampleStateCreateInfo vkinit::multisamplingStateCreateInfo()
{
	VkPipelineMultisampleStateCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;

	info.sampleShadingEnable = VK_FALSE;
	//multisampling defaulted to no multisampling (1 sample per pixel)
	info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	info.minSampleShading = 1.0f;
	info.pSampleMask = nullptr;
	info.alphaToCoverageEnable = VK_FALSE;
	info.alphaToOneEnable = VK_FALSE;

	return info;
}

VkPipelineColorBlendAttachmentState vkinit::colorBlendAttachmentState()
{
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.colorWriteMask =
		VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	colorBlendAttachment.blendEnable = VK_FALSE;

	return colorBlendAttachment;
}

VkPipelineLayoutCreateInfo vkinit::pipelineLayoutCreateInfo()
{
	VkPipelineLayoutCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

	//empty defaults
	info.flags = 0;
	info.setLayoutCount = 0;
	info.pSetLayouts = nullptr;
	info.pushConstantRangeCount = 0;
	info.pPushConstantRanges = nullptr;

	return info;
}

VkRenderPassBeginInfo vkinit::renderPassBeginInfo(const VkRenderPass& renderPass, const VkExtent2D& windowExtent,
                                                  const VkFramebuffer& framebuffer)
{
	VkRenderPassBeginInfo info{};
	info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	info.renderPass = renderPass;
	info.renderArea.offset.x = 0;
	info.renderArea.offset.y = 0;
	info.renderArea.extent = windowExtent;
	info.framebuffer = framebuffer;

	return info;
}

VkImageCreateInfo vkinit::imageCreateInfo(const VkFormat format, const VkImageUsageFlags usageFlags,
                                          const VkExtent3D extent)
{
	VkImageCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	info.imageType = VK_IMAGE_TYPE_2D;
	info.format = format;
	info.extent = extent;
	info.mipLevels = 1;
	info.arrayLayers = 1;
	info.samples = VK_SAMPLE_COUNT_1_BIT;
	info.tiling = VK_IMAGE_TILING_OPTIMAL;
	info.usage = usageFlags;

	return info;
}

VkImageViewCreateInfo vkinit::imageviewCreateInfo(const VkFormat format, const VkImage image,
                                                  const VkImageAspectFlags aspectFlags)
{
	VkImageViewCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;

	info.viewType = VK_IMAGE_VIEW_TYPE_2D;
	info.image = image;
	info.format = format;
	info.subresourceRange.baseMipLevel = 0;
	info.subresourceRange.levelCount = 1;
	info.subresourceRange.baseArrayLayer = 0;
	info.subresourceRange.layerCount = 1;
	info.subresourceRange.aspectMask = aspectFlags;

	return info;
}

VkPipelineDepthStencilStateCreateInfo vkinit::depthStencilCreateInfo(const bool depthTest, const bool depthWrite,
                                                                     const VkCompareOp compareOp)
{
	VkPipelineDepthStencilStateCreateInfo info{};
	info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
	info.depthTestEnable = depthTest ? VK_TRUE : VK_FALSE;
	info.depthWriteEnable = depthWrite ? VK_TRUE : VK_FALSE;
	info.depthCompareOp = depthTest ? compareOp : VK_COMPARE_OP_ALWAYS;
	info.depthBoundsTestEnable = VK_FALSE;
	info.minDepthBounds = 0.0f; // Optional
	info.maxDepthBounds = 1.0f; // Optional
	info.stencilTestEnable = VK_FALSE;

	return info;
}

// descriptors

VkDescriptorSetLayoutBinding vkinit::descriptorsetLayoutBinding(VkDescriptorType type, VkShaderStageFlags stageFlags,
                                                                uint32_t binding)
{
	VkDescriptorSetLayoutBinding setbind{};
	setbind.binding = binding;
	setbind.descriptorCount = 1;
	setbind.descriptorType = type;
	setbind.pImmutableSamplers = nullptr;
	setbind.stageFlags = stageFlags;

	return setbind;
}

VkWriteDescriptorSet vkinit::writeDescriptorBuffer(const VkDescriptorType type, const VkDescriptorSet dstSet,
                                                   VkDescriptorBufferInfo* const bufferInfo, const uint32_t binding)
{
	VkWriteDescriptorSet write{};
	write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

	write.dstBinding = binding;
	write.dstSet = dstSet;
	write.descriptorCount = 1;
	write.descriptorType = type;
	write.pBufferInfo = bufferInfo;

	return write;
}
