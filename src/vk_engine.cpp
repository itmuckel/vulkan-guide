#include "vk_engine.h"

#pragma warning(push, 0)

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#pragma warning(pop)

#include <fstream>
#include <iostream>
#include <algorithm>
#include <glm/gtx/transform.hpp>
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
#include "vk_initializers.h"
#include "vk_types.h"
#include "vk_textures.h"


void vkCheck(const VkResult x)
{
	if (x != VK_SUCCESS)
	{
		std::cout << "Detected Vulkan error: " << x << '\n';
		std::abort();
	}
}

const float Camera::MOVE_SPEED = 0.05f;
const float Camera::X_SPEED = 0.6f;
const float Camera::Y_SPEED = 0.6f;

void Camera::calculateDirection(const float deltaYaw, const float deltaPitch)
{
	yaw += deltaYaw;
	pitch -= deltaPitch;

	pitch = std::clamp(pitch, -89.f, 89.f);

	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

	direction = glm::normalize(direction);
}

Material* VulkanEngine::createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name)
{
	Material material{};
	material.pipeline = pipeline;
	material.pipelineLayout = layout;
	materials[name] = material;

	return &materials[name];
}

Material* VulkanEngine::getMaterial(const std::string& name)
{
	const auto it = materials.find(name);
	if (it == materials.end())
	{
		return nullptr;
	}

	return &it->second;
}

Mesh* VulkanEngine::getMesh(const std::string& name)
{
	const auto it = meshes.find(name);
	if (it == meshes.end())
	{
		return nullptr;
	}

	return &it->second;
}

void VulkanEngine::loadImages()
{
	Texture lostEmpire{};

	vkutil::loadImageFromFile(*this, "../assets/lost_empire-RGBA.png", lostEmpire.image);

	auto imageInfo = vkinit::imageviewCreateInfo(VK_FORMAT_R8G8B8A8_UNORM, lostEmpire.image.image,
	                                             VK_IMAGE_ASPECT_COLOR_BIT);
	vkCreateImageView(device, &imageInfo, nullptr, &lostEmpire.imageView);

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyImageView(device, lostEmpire.imageView, nullptr);
	});

	loadedTextures["empire_diffuse"] = lostEmpire;
}

AllocatedBuffer VulkanEngine::createBuffer(const size_t allocSize, const VkBufferUsageFlags usage,
                                           const VmaMemoryUsage memoryUsage) const
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

	bufferInfo.size = allocSize;
	bufferInfo.usage = usage;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = memoryUsage;

	AllocatedBuffer newBuffer{};
	vkCheck(vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocInfo,
	                        &newBuffer.buffer, &newBuffer.allocation, nullptr));

	return newBuffer;
}

void VulkanEngine::initDescriptors()
{
	descriptorAllocator = DescriptorAllocator{};
	descriptorAllocator.init(device);

	descriptorLayoutCache = DescriptorLayoutCache{};
	descriptorLayoutCache.init(device);

	mainDeletionQueue.pushFunction([=]()
	{
		descriptorAllocator.cleanup();
		descriptorLayoutCache.cleanup();
	});


	const auto sceneParamBufferSize = FRAME_OVERLAP * padUniformBufferSize(sizeof(GpuSceneData));
	sceneParametersBuffer = createBuffer(sceneParamBufferSize,
	                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
	                                     VMA_MEMORY_USAGE_CPU_TO_GPU);
	mainDeletionQueue.pushFunction([=]()
	{
		vmaDestroyBuffer(allocator, sceneParametersBuffer.buffer, sceneParametersBuffer.allocation);
	});

	const auto cameraBind = vkinit::descriptorsetLayoutBinding(
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		VK_SHADER_STAGE_VERTEX_BIT,
		0);

	const auto sceneBind = vkinit::descriptorsetLayoutBinding(
		VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
		1);

	VkDescriptorSetLayoutBinding bindings[] = {cameraBind, sceneBind};

	VkDescriptorSetLayoutCreateInfo setInfo{};
	setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	setInfo.bindingCount = 2;
	setInfo.flags = 0;
	setInfo.pBindings = bindings;

	const auto objectBind = vkinit::descriptorsetLayoutBinding(
		VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		VK_SHADER_STAGE_VERTEX_BIT,
		0
	);

	VkDescriptorSetLayoutCreateInfo objectSetInfo{};
	objectSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	objectSetInfo.bindingCount = 1;
	objectSetInfo.flags = 0;
	objectSetInfo.pBindings = &objectBind;

	const auto textureBind = vkinit::descriptorsetLayoutBinding(
		VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		VK_SHADER_STAGE_FRAGMENT_BIT,
		0
	);

	VkDescriptorSetLayoutCreateInfo textureSetInfo{};
	textureSetInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	textureSetInfo.bindingCount = 1;
	textureSetInfo.flags = 0;
	textureSetInfo.pBindings = &textureBind;

	globalSetLayout = descriptorLayoutCache.createDescriptorLayout(&setInfo);
	objectSetLayout = descriptorLayoutCache.createDescriptorLayout(&objectSetInfo);
	singleTextureSetLayout = descriptorLayoutCache.createDescriptorLayout(&textureSetInfo);

	for (auto i = 0; i < FRAME_OVERLAP; i += 1)
	{
		const auto maxObjects = 10000;
		frames[i].objectBuffer = createBuffer(sizeof(GpuObjectData) * maxObjects,
		                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		                                      VMA_MEMORY_USAGE_CPU_TO_GPU);
		frames[i].cameraBuffer = createBuffer(sizeof(GpuCameraData),
		                                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		                                      VMA_MEMORY_USAGE_CPU_TO_GPU);

		mainDeletionQueue.pushFunction([=]()
		{
			vmaDestroyBuffer(allocator, frames[i].objectBuffer.buffer, frames[i].objectBuffer.allocation);
			vmaDestroyBuffer(allocator, frames[i].cameraBuffer.buffer, frames[i].cameraBuffer.allocation);
		});

		descriptorAllocator.allocate(&frames[i].globalDescriptor, globalSetLayout);
		descriptorAllocator.allocate(&frames[i].objectDescriptor, objectSetLayout);

		VkDescriptorBufferInfo cameraInfo{};
		cameraInfo.buffer = frames[i].cameraBuffer.buffer;
		cameraInfo.offset = 0;
		cameraInfo.range = sizeof(GpuCameraData);

		VkDescriptorBufferInfo sceneInfo{};
		sceneInfo.buffer = sceneParametersBuffer.buffer;
		sceneInfo.offset = 0;
		sceneInfo.range = sizeof(GpuSceneData);

		VkDescriptorBufferInfo objectBufferInfo{};
		objectBufferInfo.buffer = frames[i].objectBuffer.buffer;
		objectBufferInfo.offset = 0;
		objectBufferInfo.range = sizeof(GpuObjectData) * maxObjects;

		DescriptorBuilder::begin(&descriptorLayoutCache, &descriptorAllocator)
			.bindBuffer(0, &cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, setInfo.pBindings[0].stageFlags)
			.bindBuffer(1, &sceneInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, setInfo.pBindings[1].stageFlags)
			.build(frames[i].globalDescriptor);

		DescriptorBuilder::begin(&descriptorLayoutCache, &descriptorAllocator)
			.bindBuffer(0, &objectBufferInfo, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, objectSetInfo.pBindings[0].stageFlags)
			.build(frames[i].objectDescriptor);
	}
}

void VulkanEngine::immediateSubmit(std::function<void(VkCommandBuffer vmd)>&& function) const
{
	auto cmdAllocInfo = vkinit::commandBufferAllocateInfo(uploadContext.commandPool);

	VkCommandBuffer cmd{};
	vkCheck(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd));

	const auto cmdBeginInfo = vkinit::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	vkCheck(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	function(cmd);

	vkCheck(vkEndCommandBuffer(cmd));

	auto submit = vkinit::submitInfo(&cmd);

	// uploadFence will block until the graphic commands finish execution
	vkCheck(vkQueueSubmit(graphicsQueue, 1, &submit, uploadContext.uploadFence));

	vkWaitForFences(device, 1, &uploadContext.uploadFence, true, UINT64_MAX);
	vkResetFences(device, 1, &uploadContext.uploadFence);

	// frees command buffer too
	vkResetCommandPool(device, uploadContext.commandPool, 0);
}

void VulkanEngine::init()
{
	// We initialize SDL and create a window with it.
	SDL_Init(SDL_INIT_VIDEO);

	constexpr auto windowFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI;

	window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowExtent.width,
	                          windowExtent.height, windowFlags);

	initVulkan();

	initSwapchain();

	initCommands();

	initDefaultRenderpass();

	initFramebuffers();

	initSyncStructures();

	initDescriptors();

	initPipelines();

	loadImages();

	loadMeshes();

	initScene();

	// everything went fine
	isInitialized = true;
}

void VulkanEngine::initScene()
{
	auto samplerInfo = vkinit::samplerCreateInfo(VK_FILTER_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);

	VkSampler blockySampler{};
	vkCreateSampler(device, &samplerInfo, nullptr, &blockySampler);

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroySampler(device, blockySampler, nullptr);
	});

	auto texturedMaterial = getMaterial("texturedmesh");

	descriptorAllocator.allocate(&texturedMaterial->textureSet, singleTextureSetLayout);

	VkDescriptorImageInfo imageBufferInfo{};
	imageBufferInfo.sampler = blockySampler;
	imageBufferInfo.imageView = loadedTextures["empire_diffuse"].imageView;
	imageBufferInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

	DescriptorBuilder::begin(&descriptorLayoutCache, &descriptorAllocator)
		.bindImage(0, &imageBufferInfo, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT)
		.build(texturedMaterial->textureSet);

	RenderObject map{};
	map.mesh = getMesh("empire");
	map.material = getMaterial("texturedmesh");
	map.transformMatrix = glm::translate(glm::vec3{5, -10, 0});
	renderables.push_back(map);

	RenderObject monkey{};
	monkey.mesh = getMesh("monkey");
	monkey.material = getMaterial("defaultmesh");
	monkey.transformMatrix = glm::mat4{1.f};

	renderables.push_back(monkey);

	for (auto x = -20; x <= 20; x++)
	{
		for (auto y = -20; y <= 20; y++)
		{
			RenderObject tri{};
			tri.mesh = getMesh("monkey");
			tri.material = getMaterial("monochrome");
			const auto translation = glm::translate(glm::mat4{1.0}, glm::vec3{x, 0.0, y});
			const auto scale = glm::scale(glm::mat4{1.0}, glm::vec3{0.2, 0.2, 0.2});
			tri.transformMatrix = translation * scale;

			renderables.push_back(tri);
		}
	}
}

void VulkanEngine::initVulkan()
{
	vkb::InstanceBuilder builder{};

	auto instRet = builder.set_app_name("Vulkan")
	                      .request_validation_layers(true)
	                      .require_api_version(1, 1, 0)
	                      .use_default_debug_messenger()
	                      .build();

	auto vkbInst = instRet.value();

	instance = vkbInst.instance;
	debugMessenger = vkbInst.debug_messenger;

	SDL_Vulkan_CreateSurface(window, instance, &surface);

	vkb::PhysicalDeviceSelector selector{vkbInst};
	vkb::PhysicalDevice physicalDevice = selector
	                                     .set_minimum_version(1, 1)
	                                     .set_surface(surface)
	                                     .select()
	                                     .value();

	vkb::DeviceBuilder deviceBuilder{physicalDevice};
	auto vkbDevice = deviceBuilder.build().value();

	device = vkbDevice.device;
	chosenGpu = physicalDevice.physical_device;
	vkGetPhysicalDeviceProperties(chosenGpu, &gpuProperties);
	std::cout << "The GPU has a minimum buffer alignment of "
		<< gpuProperties.limits.minUniformBufferOffsetAlignment
		<< '\n';

	graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	// initialize memory allocator

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = chosenGpu;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	vmaCreateAllocator(&allocatorInfo, &allocator);
}

size_t VulkanEngine::padUniformBufferSize(const size_t originalSize) const
{
	const auto minUboAlignment = gpuProperties.limits.minUniformBufferOffsetAlignment;
	auto alignedSize = originalSize;
	if (minUboAlignment > 0)
	{
		alignedSize = (alignedSize + minUboAlignment - 1) & ~(minUboAlignment - 1);
	}

	return alignedSize;
}

void VulkanEngine::initCommands()
{
	const auto uploadCommandPoolInfo = vkinit::commandPoolCreateInfo(graphicsQueueFamily);
	vkCheck(vkCreateCommandPool(device, &uploadCommandPoolInfo, nullptr, &uploadContext.commandPool));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyCommandPool(device, uploadContext.commandPool, nullptr);
	});

	// create a command pool for commands submitted to the graphics queue.
	// we also want the pool to allow for resetting of individual command buffers
	const auto commandPoolInfo = vkinit::commandPoolCreateInfo(
		graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	for (auto& frame : frames)
	{
		vkCheck(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &frame.commandPool));

		const auto cmdAllocInfo = vkinit::commandBufferAllocateInfo(frame.commandPool, 1);

		vkCheck(vkAllocateCommandBuffers(device, &cmdAllocInfo, &frame.mainCommandBuffer));

		mainDeletionQueue.pushFunction([=]()
		{
			vkDestroyCommandPool(device, frame.commandPool, nullptr);
		});
	}
}

void VulkanEngine::initDefaultRenderpass()
{
	// the renderpass will use this color attachment.
	VkAttachmentDescription colorAttachment{};
	colorAttachment.format = swapchainImageFormat;
	//1 sample, we wont be doing MSAA
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//we don't care about stencil
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//we don't know or care about the starting layout of the attachment
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//after the renderpass ends, the image has to be on a layout ready for display
	// TODO: COLOR_ATTACHMENT_OPTIMAL?
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	//attachment number will index into the pAttachments array in the parent renderpass itself
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	VkAttachmentDescription depthAttachment{};
	depthAttachment.format = depthFormat;
	depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
	depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

	VkAttachmentReference depthAttachmentRef{};
	depthAttachmentRef.attachment = 1;
	depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;


	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;
	subpass.pDepthStencilAttachment = &depthAttachmentRef;

	VkAttachmentDescription attachments[2] = {colorAttachment, depthAttachment};

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	//connect the color attachment to the info
	renderPassInfo.attachmentCount = 2;
	renderPassInfo.pAttachments = attachments;
	//connect the subpass to the info
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;

	vkCheck(vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyRenderPass(device, renderPass, nullptr);
	});
}

void VulkanEngine::initFramebuffers()
{
	//create the framebuffers for the swapchain images. This will connect the render-pass to the images for rendering
	VkFramebufferCreateInfo fbInfo{};
	fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
	fbInfo.renderPass = renderPass;
	fbInfo.width = windowExtent.width;
	fbInfo.height = windowExtent.height;
	fbInfo.layers = 1;

	//grab how many images we have in the swapchain
	const auto swapchainImagecount = swapchainImages.size();
	framebuffers = std::vector<VkFramebuffer>(swapchainImagecount);

	//create framebuffers for each of the swapchain image views
	for (size_t i = 0; i < swapchainImagecount; i += 1)
	{
		VkImageView attachments[2] = {swapchainImageViews[i], depthImageView};
		fbInfo.attachmentCount = 2;
		fbInfo.pAttachments = attachments;

		vkCheck(vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffers[i]));

		mainDeletionQueue.pushFunction([=]()
		{
			vkDestroyFramebuffer(device, framebuffers[i], nullptr);
			vkDestroyImageView(device, swapchainImageViews[i], nullptr);
		});
	}
}

void VulkanEngine::initSyncStructures()
{
	VkFenceCreateInfo uploadFenceCreateInfo{};
	uploadFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	//we want to create the fence with the Create  Signaled flag, so we can wait on it before using it on a gpu command (for the first frame)
	uploadFenceCreateInfo.flags = 0;

	vkCheck(vkCreateFence(device, &uploadFenceCreateInfo, nullptr, &uploadContext.uploadFence));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyFence(device, uploadContext.uploadFence, nullptr);
	});

	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	//we want to create the fence with the Create  Signaled flag, so we can wait on it before using it on a gpu command (for the first frame)
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	for (auto& frame : frames)
	{
		vkCheck(vkCreateFence(device, &fenceCreateInfo, nullptr, &frame.renderFence));

		mainDeletionQueue.pushFunction([=]()
		{
			vkDestroyFence(device, frame.renderFence, nullptr);
		});

		VkSemaphoreCreateInfo semaphoreCreateInfo{};
		semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		vkCheck(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.presentSemaphore));
		vkCheck(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &frame.renderSemaphore));

		mainDeletionQueue.pushFunction([=]()
		{
			vkDestroySemaphore(device, frame.presentSemaphore, nullptr);
			vkDestroySemaphore(device, frame.renderSemaphore, nullptr);
		});
	}
}

bool VulkanEngine::loadShaderModule(const char* filePath, VkShaderModule& outShaderModule) const
{
	//open the file. With cursor at the end
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);

	if (!file.is_open())
	{
		return false;
	}

	//find what the size of the file is by looking up the location of the cursor
	//because the cursor is at the end, it gives the size directly in bytes
	const auto fileSize = static_cast<size_t>(file.tellg());

	//spirv expects the buffer to be on uint32, so make sure to reserve a int vector big enough for the entire file
	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	//put file cursor at beggining
	file.seekg(0);

	//load the entire file into the buffer
	file.read(reinterpret_cast<char*>(buffer.data()), fileSize);

	//now that the file is loaded into the buffer, we can close it
	file.close();


	//create a new shader module, using the buffer we loaded
	VkShaderModuleCreateInfo createInfo{};
	createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

	//codeSize has to be in bytes, so multply the ints in the buffer by size of int to know the real size of the buffer
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	//check that the creation goes well.
	VkShaderModule shaderModule;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
	{
		return false;
	}
	outShaderModule = shaderModule;
	return true;
}

void VulkanEngine::initPipelines()
{
	VkShaderModule meshVert{};
	if (!loadShaderModule("../shaders/mesh.vert.spv", meshVert))
	{
		std::cout << "Error when building the mesh vertex shader module" << std::endl;
	}

	VkShaderModule defaultLitFrag{};
	if (!loadShaderModule("../shaders/default_lit.frag.spv", defaultLitFrag))
	{
		std::cout << "Error when building the default_lit fragment shader module" << std::endl;
	}

	VkShaderModule texturedMeshFrag{};
	if (!loadShaderModule("../shaders/textured_lit.frag.spv", texturedMeshFrag))
	{
		std::cout << "Error when building the textured_lit fragment shader module" << std::endl;
	}

	VkShaderModule meshFrag{};
	if (!loadShaderModule("../shaders/mesh.frag.spv", meshFrag))
	{
		std::cout << "Error when building the mesh fragment shader module" << std::endl;
	}

	// build the pipeline layout that controls the inputs/outputs of the shader
	auto pipeline_layout_info = vkinit::pipelineLayoutCreateInfo();

	VkDescriptorSetLayout set_layouts[] = {globalSetLayout, objectSetLayout};
	pipeline_layout_info.setLayoutCount = 2;
	pipeline_layout_info.pSetLayouts = set_layouts;

	VkPipelineLayout monochrome_pipeline_layout{};

	vkCheck(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &monochrome_pipeline_layout));

	PipelineBuilder pipeline_builder{};

	// vertex input controls how to read vertices from vertex buffers. We arent using it yet
	pipeline_builder.vertexInputInfo = vkinit::vertexInputStateCreateInfo();

	auto vertex_description = Vertex::getVertexDescription();

	// connect the pipeline builder vertex input info to the one we get from Vertex
	pipeline_builder.vertexInputInfo.pVertexAttributeDescriptions = vertex_description.attributes.data();
	pipeline_builder.vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(
		vertex_description.attributes.size());

	pipeline_builder.vertexInputInfo.pVertexBindingDescriptions = vertex_description.bindings.data();
	pipeline_builder.vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(
		vertex_description.bindings.size());

	pipeline_builder.inputAssembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
	pipeline_builder.viewport = {
		0.0f, 0.0f,
		static_cast<float>(windowExtent.width), static_cast<float>(windowExtent.height),
		0.0f, 1.0f
	};
	pipeline_builder.scissor = {{0, 0}, windowExtent};
	pipeline_builder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);
	pipeline_builder.multisampling = vkinit::multisamplingStateCreateInfo();
	pipeline_builder.colorBlendAttachment = vkinit::colorBlendAttachmentState();
	pipeline_builder.depthStencil = vkinit::depthStencilCreateInfo(true, true, VK_COMPARE_OP_LESS_OR_EQUAL);


	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVert));

	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, defaultLitFrag));

	// use the triangle layout we created
	pipeline_builder.pipelineLayout = monochrome_pipeline_layout;

	// finally build the pipeline
	const auto monochrome_pipeline = pipeline_builder.buildPipeline(device, renderPass);
	createMaterial(monochrome_pipeline, monochrome_pipeline_layout, "monochrome");

	// build the mesh pipeline

	// clear the shader stages for the builder
	pipeline_builder.shaderStages.clear();

	// add the other shaders
	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVert));

	// make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, meshFrag));

	// set up push constants
	auto meshPipelineLayoutInfo = vkinit::pipelineLayoutCreateInfo();

	//meshPipelineLayoutInfo.pushConstantRangeCount = 1;
	//meshPipelineLayoutInfo.pPushConstantRanges = &push_constant;

	meshPipelineLayoutInfo.setLayoutCount = 2;
	meshPipelineLayoutInfo.pSetLayouts = set_layouts;

	VkPipelineLayout meshPipelineLayout{};

	vkCheck(vkCreatePipelineLayout(device, &meshPipelineLayoutInfo, nullptr, &meshPipelineLayout));

	pipeline_builder.pipelineLayout = meshPipelineLayout;

	// build the mesh triangle pipeline
	const auto meshPipeline = pipeline_builder.buildPipeline(device, renderPass);

	createMaterial(meshPipeline, meshPipelineLayout, "defaultmesh");

	auto texturedPipelineLayoutInfo = meshPipelineLayoutInfo;

	VkDescriptorSetLayout texturedSetLayouts[] = {globalSetLayout, objectSetLayout, singleTextureSetLayout};

	texturedPipelineLayoutInfo.setLayoutCount = 3;
	texturedPipelineLayoutInfo.pSetLayouts = texturedSetLayouts;

	VkPipelineLayout textured_pipeline_layout{};
	vkCheck(vkCreatePipelineLayout(device, &texturedPipelineLayoutInfo, nullptr, &textured_pipeline_layout));

	pipeline_builder.shaderStages.clear();
	// add the other shaders
	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVert));

	// make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipeline_builder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, texturedMeshFrag));

	pipeline_builder.pipelineLayout = textured_pipeline_layout;
	const auto texPipeline = pipeline_builder.buildPipeline(device, renderPass);

	createMaterial(texPipeline, textured_pipeline_layout, "texturedmesh");

	// cleanup

	// deleting all of the vulkan shaders
	vkDestroyShaderModule(device, meshVert, nullptr);
	vkDestroyShaderModule(device, meshFrag, nullptr);
	vkDestroyShaderModule(device, texturedMeshFrag, nullptr);
	vkDestroyShaderModule(device, defaultLitFrag, nullptr);

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyPipeline(device, monochrome_pipeline, nullptr);
		vkDestroyPipelineLayout(device, monochrome_pipeline_layout, nullptr);
		vkDestroyPipeline(device, meshPipeline, nullptr);
		vkDestroyPipeline(device, texPipeline, nullptr);
		vkDestroyPipelineLayout(device, meshPipelineLayout, nullptr);
		vkDestroyPipelineLayout(device, textured_pipeline_layout, nullptr);
	});
}

void VulkanEngine::loadMeshes()
{
	Mesh triangleMesh{};
	triangleMesh.vertices.resize(3);

	triangleMesh.vertices[0].position = {1.f, 1.f, 0.f};
	triangleMesh.vertices[1].position = {-1.f, 1.f, 0.f};
	triangleMesh.vertices[2].position = {0.f, -1.f, 0.f};

	triangleMesh.vertices[0].color = {0.f, 1.f, 0.f};
	triangleMesh.vertices[1].color = {0.f, 1.f, 0.f};
	triangleMesh.vertices[2].color = {0.f, 1.f, 0.f};

	uploadMesh(triangleMesh);

	Mesh monkeyMesh{};
	monkeyMesh.loadFromObj("../assets/monkey_smooth.obj");
	uploadMesh(monkeyMesh);

	Mesh lostEmpire{};
	lostEmpire.loadFromObj("../assets/lost_empire.obj");
	uploadMesh(lostEmpire);

	meshes["empire"] = lostEmpire;
	meshes["monkey"] = monkeyMesh;
	meshes["triangle"] = triangleMesh;
}

void VulkanEngine::uploadMesh(Mesh& mesh)
{
	// create CPU source buffer
	VkBufferCreateInfo stagingBufferInfo{};
	stagingBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	stagingBufferInfo.size = mesh.vertices.size() * sizeof(Vertex);
	stagingBufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;

	AllocatedBuffer stagingBuffer{};

	vkCheck(vmaCreateBuffer(allocator, &stagingBufferInfo, &vmaAllocInfo,
	                        &stagingBuffer.buffer,
	                        &stagingBuffer.allocation,
	                        nullptr));

	// copy the vertex data
	void* data{};
	vmaMapMemory(allocator, stagingBuffer.allocation, &data);
	memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
	vmaUnmapMemory(allocator, stagingBuffer.allocation);

	// create GPU only buffer
	VkBufferCreateInfo vertexBufferInfo{};
	vertexBufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferInfo.size = stagingBufferInfo.size;
	vertexBufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

	vmaAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

	vkCheck(vmaCreateBuffer(allocator, &vertexBufferInfo, &vmaAllocInfo,
	                        &mesh.vertexBuffer.buffer,
	                        &mesh.vertexBuffer.allocation,
	                        nullptr));

	// copy
	immediateSubmit([=](VkCommandBuffer cmd)
	{
		VkBufferCopy copy{};
		copy.srcOffset = 0;
		copy.dstOffset = 0;
		copy.size = stagingBufferInfo.size;

		vkCmdCopyBuffer(cmd, stagingBuffer.buffer, mesh.vertexBuffer.buffer,
		                1, &copy);
	});

	mainDeletionQueue.pushFunction([=]()
	{
		vmaDestroyBuffer(allocator, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);
	});

	vmaDestroyBuffer(allocator, stagingBuffer.buffer, stagingBuffer.allocation);
}

VkPipeline PipelineBuilder::buildPipeline(const VkDevice device, const VkRenderPass pass)
{
	// make viewport state from our stored viewport and scissor.
	// at the moment we wont support multiple viewports or scissors
	VkPipelineViewportStateCreateInfo viewportState{};
	viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;

	viewportState.viewportCount = 1;
	viewportState.pViewports = &viewport;
	viewportState.scissorCount = 1;
	viewportState.pScissors = &scissor;

	// setup dummy color blending. We arent using transparent objects yet
	// the blending is just "no blend", but we do write to the color attachment
	VkPipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

	colorBlending.logicOpEnable = VK_FALSE;
	colorBlending.logicOp = VK_LOGIC_OP_COPY;
	colorBlending.attachmentCount = 1;
	colorBlending.pAttachments = &colorBlendAttachment;

	// build the actual pipeline
	// we now use all of the info structs we have been writing into into this one to create the pipeline
	VkGraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;

	pipelineInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.pDepthStencilState = &depthStencil;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = nullptr;

	// its easy to error out on create graphics pipeline, so we handle it a bit better than the common vkCheck case
	VkPipeline new_pipeline{};
	if (vkCreateGraphicsPipelines(
		device, nullptr, 1, &pipelineInfo, nullptr, &new_pipeline) != VK_SUCCESS)
	{
		std::cout << "failed to create pipline\n";
		return nullptr; // failed to create graphics pipeline
	}

	return new_pipeline;
}

void VulkanEngine::initSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{chosenGpu, device, surface};

	vkb::Swapchain vkbSwapchain = swapchainBuilder
	                              .use_default_format_selection()
	                              .set_desired_present_mode(VK_PRESENT_MODE_MAILBOX_KHR)
	                              .set_desired_extent(windowExtent.width, windowExtent.height)
	                              .build().value();

	swapchain = vkbSwapchain.swapchain;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchainImageFormat = vkbSwapchain.image_format;

	// depth image

	const VkExtent3D depthImageExtent{windowExtent.width, windowExtent.height, 1};
	depthFormat = VK_FORMAT_D32_SFLOAT;
	auto depthImageInfo = vkinit::imageCreateInfo(depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
	                                              depthImageExtent);
	VmaAllocationCreateInfo depthImageAllocInfo{};
	depthImageAllocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
	depthImageAllocInfo.requiredFlags = static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

	vmaCreateImage(allocator, &depthImageInfo, &depthImageAllocInfo, &depthImage.image, &depthImage.allocation,
	               nullptr);
	auto depthImageViewInfo = vkinit::imageviewCreateInfo(depthFormat, depthImage.image, VK_IMAGE_ASPECT_DEPTH_BIT);

	vkCheck(vkCreateImageView(device, &depthImageViewInfo, nullptr, &depthImageView));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyImageView(device, depthImageView, nullptr);
		vmaDestroyImage(allocator, depthImage.image, depthImage.allocation);

		vkDestroySwapchainKHR(device, swapchain, nullptr);
	});
}

void VulkanEngine::cleanup()
{
	if (isInitialized)
	{
		vkDeviceWaitIdle(device);

		mainDeletionQueue.flush();

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkb::destroy_debug_utils_messenger(instance, debugMessenger, nullptr);
		vmaDestroyAllocator(allocator);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		SDL_DestroyWindow(window);
	}
}

void VulkanEngine::draw()
{
	// wait until the GPU has finished rendering the last frame
	vkCheck(vkWaitForFences(device, 1, &getCurrentFrame().renderFence, true, UINT64_MAX));
	vkCheck(vkResetFences(device, 1, &getCurrentFrame().renderFence));

	// request image from the swapchain
	uint32_t swapchainImageIndex{};
	vkCheck(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, getCurrentFrame().presentSemaphore,
	                              nullptr, &swapchainImageIndex));

	// now that we are sure that the commands finished executing,
	// we can safely reset the command buffer to begin recording again.
	vkCheck(vkResetCommandBuffer(getCurrentFrame().mainCommandBuffer, 0));

	// naming it cmd for shorter writing
	auto cmd = getCurrentFrame().mainCommandBuffer;

	// begin the command buffer recording. We will use this command buffer exactly once,
	// so we want to let Vulkan know that
	const auto cmdBeginInfo = vkinit::commandBufferBeginInfo(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

	vkCheck(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	const auto flash = abs(sin(frameNumber / 60.f));
	const VkClearValue clearValue =
		{{{0.0f, 0.0f, flash, 1.0f}}};

	VkClearValue depthClear{};
	depthClear.depthStencil.depth = 1.f;

	// start the main renderpass
	// We will use the clear color from above and
	// the framebuffer of the index the swapchain gave us

	auto rpInfo = vkinit::renderPassBeginInfo(renderPass, windowExtent, framebuffers[swapchainImageIndex]);

	rpInfo.clearValueCount = 2;
	VkClearValue clearValues[] = {clearValue, depthClear};
	rpInfo.pClearValues = clearValues;

	// -------------- begin draw -----------------------------------

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	drawObjects(cmd, renderables.data(), renderables.size());

	vkCmdEndRenderPass(cmd);
	vkCheck(vkEndCommandBuffer(cmd));

	// -------------- end draw -------------------------------------

	// we want to wait on the presentSemaphore, as that semaphore is signaled
	// when the swapchain is ready
	// we will signal the renderSemaphore, to signal that rendering has finished

	auto submit = vkinit::submitInfo(&cmd);

	constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &getCurrentFrame().presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &getCurrentFrame().renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	// submit command buffer to the queue and execute it.
	// renderFence will now block until the graphic commands finish execution
	vkCheck(vkQueueSubmit(graphicsQueue, 1, &submit, getCurrentFrame().renderFence));

	// this will put the image we just rendered to into the visible window.
	// we want to wait on the renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &getCurrentFrame().renderSemaphore;

	presentInfo.pImageIndices = &swapchainImageIndex;

	vkCheck(vkQueuePresentKHR(graphicsQueue, &presentInfo));

	//increase the number of frames drawn
	frameNumber++;
}

const FrameData& VulkanEngine::getCurrentFrame() const
{
	return frames[frameNumber % FRAME_OVERLAP];
}

void VulkanEngine::drawObjects(const VkCommandBuffer cmd, RenderObject* first, int count)
{
	// camera

	const glm::mat4 view = glm::lookAt(camera.pos, camera.pos + camera.direction, camera.up);
	glm::mat4 projection = glm::perspective(glm::radians(70.f),
	                                        static_cast<float>(windowExtent.width) / windowExtent.height,
	                                        0.1f, 200.0f);
	projection[1][1] *= -1;

	GpuCameraData camData{};
	camData.projection = projection;
	camData.view = view;
	camData.viewproj = projection * view;

	void* data;
	vmaMapMemory(allocator, getCurrentFrame().cameraBuffer.allocation, &data);
	memcpy(data, &camData, sizeof(GpuCameraData));
	vmaUnmapMemory(allocator, getCurrentFrame().cameraBuffer.allocation);

	// scene

	const auto framed = frameNumber / 60.f;
	sceneParameters.ambientColor = glm::vec4(sin(framed), 0, cos(framed), 1);

	char* sceneData{};
	vmaMapMemory(allocator, sceneParametersBuffer.allocation, reinterpret_cast<void**>(&sceneData));

	const auto frameIndex = frameNumber % FRAME_OVERLAP;

	sceneData += padUniformBufferSize(sizeof(GpuSceneData)) * frameIndex;
	memcpy(sceneData, &sceneParameters, sizeof(GpuSceneData));

	vmaUnmapMemory(allocator, sceneParametersBuffer.allocation);


	// objects

	void* objectData{};
	vmaMapMemory(allocator, getCurrentFrame().objectBuffer.allocation, &objectData);

	const auto objectSsbo = static_cast<GpuObjectData*>(objectData);
	for (auto i = 0; i < count; i += 1)
	{
		auto& object = first[i];
		objectSsbo[i].modelMatrix = object.transformMatrix;
	}

	vmaUnmapMemory(allocator, getCurrentFrame().objectBuffer.allocation);

	Mesh* lastMesh = nullptr;
	Material* lastMaterial = nullptr;
	for (auto i = 0; i < count; i += 1)
	{
		auto& object = first[i];

		if (object.material == nullptr || object.mesh == nullptr)
		{
			continue;
		}

		//only bind the pipeline if it doesnt match with the already bound one
		if (object.material != lastMaterial)
		{
			vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipeline);
			lastMaterial = object.material;

			// camera data descriptor
			const uint32_t uniformOffset = padUniformBufferSize(sizeof(GpuSceneData)) * frameIndex;
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 0, 1,
			                        &getCurrentFrame().globalDescriptor, 1, &uniformOffset);

			// object data descriptor
			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 1, 1,
			                        &getCurrentFrame().objectDescriptor, 0, nullptr);

			if (object.material->textureSet != nullptr)
			{
				vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, object.material->pipelineLayout, 2, 1,
				                        &object.material->textureSet, 0, nullptr);
			}
		}


		//only bind the mesh if its a different one from last bind
		if (object.mesh != lastMesh)
		{
			//bind the mesh vertex buffer with offset 0
			VkDeviceSize offset = 0;
			vkCmdBindVertexBuffers(cmd, 0, 1, &object.mesh->vertexBuffer.buffer, &offset);
			lastMesh = object.mesh;
		}
		//we can now draw
		vkCmdDraw(cmd, object.mesh->vertices.size(), 1, 0, i);
	}
}

void VulkanEngine::run()
{
	SDL_Event e{};
	auto bQuit = false;

	SDL_SetRelativeMouseMode(SDL_TRUE);

	while (!bQuit)
	{
		const auto keystate = SDL_GetKeyboardState(nullptr);

		if (keystate[SDL_SCANCODE_W])
		{
			camera.pos += Camera::MOVE_SPEED * camera.direction;
		}
		if (keystate[SDL_SCANCODE_S])
		{
			camera.pos -= Camera::MOVE_SPEED * camera.direction;
		}
		if (keystate[SDL_SCANCODE_A])
		{
			camera.pos -= glm::normalize(glm::cross(camera.direction, camera.up)) * Camera::MOVE_SPEED;
		}
		if (keystate[SDL_SCANCODE_D])
		{
			camera.pos += glm::normalize(glm::cross(camera.direction, camera.up)) * Camera::MOVE_SPEED;
		}

		//camera.pos.y = camera.height;

		int mouseX{}, mouseY{};
		SDL_GetRelativeMouseState(&mouseX, &mouseY);

		camera.calculateDirection(mouseX * Camera::X_SPEED, mouseY * Camera::Y_SPEED);

		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT)
			{
				bQuit = true;
			}

			if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_ESCAPE)
				{
					bQuit = true;
				}
			}
		}

		draw();
	}
}
