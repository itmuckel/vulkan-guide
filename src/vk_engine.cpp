#include "vk_engine.h"

#pragma warning(push, 0)

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#pragma warning(pop)

#include <fstream>
#include <iostream>
#include "vk_initializers.h"
#include "vk_types.h"
#include <VkBootstrap.h>
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

void vkCheck(const VkResult x)
{
	if (x != VK_SUCCESS)
	{
		std::cout << "Detected Vulkan error: " << x << '\n';
		std::abort();
	}
}

void VulkanEngine::init()
{
	// We initialize SDL and create a window with it.
	SDL_Init(SDL_INIT_VIDEO);

	constexpr auto window_flags = SDL_WINDOW_VULKAN | SDL_WINDOW_ALLOW_HIGHDPI;

	window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowExtent.width,
	                          windowExtent.height, window_flags);

	initVulkan();

	initSwapchain();

	initCommands();

	initDefaultRenderpass();

	initFramebuffers();

	initSyncStructures();

	initPipelines();

	loadMeshes();

	// everything went fine
	isInitialized = true;
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

	graphicsQueue = vkbDevice.get_queue(vkb::QueueType::graphics).value();
	graphicsQueueFamily = vkbDevice.get_queue_index(vkb::QueueType::graphics).value();

	// initialize memory allocator

	VmaAllocatorCreateInfo allocatorInfo{};
	allocatorInfo.physicalDevice = chosenGpu;
	allocatorInfo.device = device;
	allocatorInfo.instance = instance;
	vmaCreateAllocator(&allocatorInfo, &allocator);
}

void VulkanEngine::initCommands()
{
	//create a command pool for commands submitted to the graphics queue.
	//we also want the pool to allow for resetting of individual command buffers
	const auto commandPoolInfo = vkinit::commandPoolCreateInfo(
		graphicsQueueFamily, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	vkCheck(vkCreateCommandPool(device, &commandPoolInfo, nullptr, &commandPool));

	//allocate the default command buffer that we will use for rendering
	const auto cmdAllocInfo = vkinit::commandBufferAllocateInfo(commandPool, 1);

	vkCheck(vkAllocateCommandBuffers(device, &cmdAllocInfo, &mainCommandBuffer));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyCommandPool(device, commandPool, nullptr);
	});
}

void VulkanEngine::initDefaultRenderpass()
{
	// the renderpass will use this color attachment.
	VkAttachmentDescription colorAttachment{};
	//the attachment will have the format needed by the swapchain
	colorAttachment.format = swapchainImageFormat;
	//1 sample, we wont be doing MSAA
	colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
	// we Clear when this attachment is loaded
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	// we keep the attachment stored when the renderpass ends
	colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
	//we don't care about stencil
	colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
	colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

	//we don't know or care about the starting layout of the attachment
	colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	//after the renderpass ends, the image has to be on a layout ready for display
	colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

	VkAttachmentReference colorAttachmentRef{};
	//attachment number will index into the pAttachments array in the parent renderpass itself
	colorAttachmentRef.attachment = 0;
	colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

	//we are going to create 1 subpass, which is the minimum you can do
	VkSubpassDescription subpass{};
	subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments = &colorAttachmentRef;

	VkRenderPassCreateInfo renderPassInfo{};
	renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;

	//connect the color attachment to the info
	renderPassInfo.attachmentCount = 1;
	renderPassInfo.pAttachments = &colorAttachment;
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
	fbInfo.attachmentCount = 1;
	fbInfo.width = windowExtent.width;
	fbInfo.height = windowExtent.height;
	fbInfo.layers = 1;

	//grab how many images we have in the swapchain
	const auto swapchainImagecount = swapchainImages.size();
	framebuffers = std::vector<VkFramebuffer>(swapchainImagecount);

	//create framebuffers for each of the swapchain image views
	for (size_t i = 0; i < swapchainImagecount; i += 1)
	{
		fbInfo.pAttachments = &swapchainImageViews[i];
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
	VkFenceCreateInfo fenceCreateInfo{};
	fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	//we want to create the fence with the Create  Signaled flag, so we can wait on it before using it on a gpu command (for the first frame)
	fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

	vkCheck(vkCreateFence(device, &fenceCreateInfo, nullptr, &renderFence));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyFence(device, renderFence, nullptr);
	});

	VkSemaphoreCreateInfo semaphoreCreateInfo{};
	semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

	vkCheck(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &presentSemaphore));
	vkCheck(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderSemaphore));

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroySemaphore(device, presentSemaphore, nullptr);
		vkDestroySemaphore(device, renderSemaphore, nullptr);
	});
}

bool VulkanEngine::loadShaderModule(const char* filePath, VkShaderModule& outShaderModule)
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
	VkShaderModule monochromeTriangleFragShader{};
	if (!loadShaderModule("../shaders/triangle.frag.spv", monochromeTriangleFragShader))
	{
		std::cout << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle fragment shader succesfully loaded" << std::endl;
	}

	VkShaderModule monochromeTriangleVertexShader{};
	if (!loadShaderModule("../shaders/triangle.vert.spv", monochromeTriangleVertexShader))
	{
		std::cout << "Error when building the triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle vertex shader succesfully loaded" << std::endl;
	}

	VkShaderModule triangleFragShader{};
	if (!loadShaderModule("../shaders/colored_triangle.frag.spv", triangleFragShader))
	{
		std::cout << "Error when building the triangle fragment shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle fragment shader succesfully loaded" << std::endl;
	}

	VkShaderModule triangleVertexShader{};
	if (!loadShaderModule("../shaders/colored_triangle.vert.spv", triangleVertexShader))
	{
		std::cout << "Error when building the triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "Triangle vertex shader succesfully loaded" << std::endl;
	}

	// build the pipeline layout that controls the inputs/outputs of the shader
	// we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
	const auto pipelineLayoutInfo = vkinit::pipelineLayoutCreateInfo();

	vkCheck(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &monochromeTrianglePipelineLayout));

	// build the stage-create-info for both vertex and fragment stages. This lets the pipeline know the shader modules per stage
	PipelineBuilder pipelineBuilder{};

	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, monochromeTriangleVertexShader));

	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, monochromeTriangleFragShader));

	// vertex input controls how to read vertices from vertex buffers. We arent using it yet
	pipelineBuilder.vertexInputInfo = vkinit::vertexInputStateCreateInfo();

	// input assembly is the configuration for drawing triangle lists, strips, or individual points.
	// we are just going to draw triangle list
	pipelineBuilder.inputAssembly = vkinit::inputAssemblyCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

	// build viewport and scissor from the swapchain extents
	pipelineBuilder.viewport.x = 0.0f;
	pipelineBuilder.viewport.y = 0.0f;
	pipelineBuilder.viewport.width = static_cast<float>(windowExtent.width);
	pipelineBuilder.viewport.height = static_cast<float>(windowExtent.height);
	pipelineBuilder.viewport.minDepth = 0.0f;
	pipelineBuilder.viewport.maxDepth = 1.0f;

	pipelineBuilder.scissor.offset = {0, 0};
	pipelineBuilder.scissor.extent = windowExtent;

	// configure the rasterizer to draw filled triangles
	pipelineBuilder.rasterizer = vkinit::rasterizationStateCreateInfo(VK_POLYGON_MODE_FILL);

	// we don't use multisampling, so just run the default one
	pipelineBuilder.multisampling = vkinit::multisamplingStateCreateInfo();

	// a single blend attachment with no blending and writing to RGBA
	pipelineBuilder.colorBlendAttachment = vkinit::colorBlendAttachmentState();

	// use the triangle layout we created
	pipelineBuilder.pipelineLayout = monochromeTrianglePipelineLayout;

	// finally build the pipeline
	monochromeTrianglePipeline = pipelineBuilder.buildPipeline(device, renderPass);

	// clear the shader stages for the builder
	pipelineBuilder.shaderStages.clear();

	// add the other shaders
	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, triangleVertexShader));

	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

	// build the red triangle pipeline
	trianglePipeline = pipelineBuilder.buildPipeline(device, renderPass);

	// build the mesh pipeline

	auto vertexDescription = Vertex::getVertexDescription();

	// connect the pipeline builder vertex input info to the one we get from Vertex
	pipelineBuilder.vertexInputInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();
	pipelineBuilder.vertexInputInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();

	pipelineBuilder.vertexInputInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
	pipelineBuilder.vertexInputInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();

	// clear the shader stages for the builder
	pipelineBuilder.shaderStages.clear();

	// compile mesh vertex shader

	VkShaderModule meshVertShader;
	if (!loadShaderModule("../shaders/tri_mesh.vert.spv", meshVertShader))
	{
		std::cout << "Error when building the triangle vertex shader module" << std::endl;
	}
	else
	{
		std::cout << "Red Triangle vertex shader succesfully loaded" << std::endl;
	}

	// add the other shaders
	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, meshVertShader));

	// make sure that triangleFragShader is holding the compiled colored_triangle.frag
	pipelineBuilder.shaderStages.push_back(
		vkinit::pipelineShaderStageCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, triangleFragShader));

	// build the mesh triangle pipeline
	meshPipeline = pipelineBuilder.buildPipeline(device, renderPass);

	// cleanup

	// deleting all of the vulkan shaders
	vkDestroyShaderModule(device, meshVertShader, nullptr);
	vkDestroyShaderModule(device, monochromeTriangleVertexShader, nullptr);
	vkDestroyShaderModule(device, monochromeTriangleFragShader, nullptr);
	vkDestroyShaderModule(device, triangleVertexShader, nullptr);
	vkDestroyShaderModule(device, triangleFragShader, nullptr);

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroyPipeline(device, monochromeTrianglePipeline, nullptr);
		vkDestroyPipeline(device, trianglePipeline, nullptr);
		vkDestroyPipeline(device, meshPipeline, nullptr);
		vkDestroyPipelineLayout(device, monochromeTrianglePipelineLayout, nullptr);
	});
}

void VulkanEngine::loadMeshes()
{
	triangleMesh.vertices.resize(3);

	triangleMesh.vertices[0].position = {1.f, 1.f, 0.f};
	triangleMesh.vertices[1].position = {-1.f, 1.f, 0.f};
	triangleMesh.vertices[2].position = {0.f, -1.f, 0.f};

	triangleMesh.vertices[0].color = {0.f, 1.f, 0.f};
	triangleMesh.vertices[1].color = {0.f, 1.f, 0.f};
	triangleMesh.vertices[2].color = {0.f, 1.f, 0.f};

	uploadMesh(triangleMesh);
}

void VulkanEngine::uploadMesh(Mesh& mesh)
{
	VkBufferCreateInfo bufferInfo{};
	bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferInfo.size = mesh.vertices.size() * sizeof(Vertex);
	bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

	VmaAllocationCreateInfo vmaAllocInfo{};
	vmaAllocInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

	vkCheck(vmaCreateBuffer(allocator, &bufferInfo, &vmaAllocInfo, &mesh.vertexBuffer.buffer,
	                        &mesh.vertexBuffer.allocation, nullptr));

	// copy the vertex data to GPU
	void* data{};
	vmaMapMemory(allocator, mesh.vertexBuffer.allocation, &data);
	memcpy(data, mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
	vmaUnmapMemory(allocator, mesh.vertexBuffer.allocation);


	mainDeletionQueue.pushFunction([=]()
	{
		vmaDestroyBuffer(allocator, mesh.vertexBuffer.buffer, mesh.vertexBuffer.allocation);
		vmaDestroyAllocator(allocator);
	});
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

	pipelineInfo.stageCount = shaderStages.size();
	pipelineInfo.pStages = shaderStages.data();
	pipelineInfo.pVertexInputState = &vertexInputInfo;
	pipelineInfo.pInputAssemblyState = &inputAssembly;
	pipelineInfo.pViewportState = &viewportState;
	pipelineInfo.pRasterizationState = &rasterizer;
	pipelineInfo.pMultisampleState = &multisampling;
	pipelineInfo.pColorBlendState = &colorBlending;
	pipelineInfo.layout = pipelineLayout;
	pipelineInfo.renderPass = pass;
	pipelineInfo.subpass = 0;
	pipelineInfo.basePipelineHandle = nullptr;

	// its easy to error out on create graphics pipeline, so we handle it a bit better than the common vkCheck case
	VkPipeline newPipeline{};
	if (vkCreateGraphicsPipelines(
		device, nullptr, 1, &pipelineInfo, nullptr, &newPipeline) != VK_SUCCESS)
	{
		std::cout << "failed to create pipline\n";
		return nullptr; // failed to create graphics pipeline
	}

	return newPipeline;
}

void VulkanEngine::initSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{chosenGpu, device, surface};

	vkb::Swapchain vkbSwapchain = swapchainBuilder
	                              .use_default_format_selection()
	                              .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	                              .set_desired_extent(windowExtent.width, windowExtent.height)
	                              .build().value();

	swapchain = vkbSwapchain.swapchain;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchainImageFormat = vkbSwapchain.image_format;

	mainDeletionQueue.pushFunction([=]()
	{
		vkDestroySwapchainKHR(device, swapchain, nullptr);
	});
}

void VulkanEngine::cleanup()
{
	if (isInitialized)
	{
		vkDeviceWaitIdle(device);

		//vkDestroySemaphore(device, renderSemaphore, nullptr);
		//vkDestroySemaphore(device, presentSemaphore, nullptr);
		//vkDestroyFence(device, renderFence, nullptr);

		//vkDestroySwapchainKHR(device, swapchain, nullptr);

		////destroy the main renderpass
		//vkDestroyRenderPass(device, renderPass, nullptr);

		////destroy swapchain resources
		//for (auto i = 0; i < framebuffers.size(); i += 1)
		//{
		//	vkDestroyFramebuffer(device, framebuffers[i], nullptr);
		//	vkDestroyImageView(device, swapchainImageViews[i], nullptr);
		//}
		//vkDestroyCommandPool(device, commandPool, nullptr);

		mainDeletionQueue.flush();

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkb::destroy_debug_utils_messenger(instance, debugMessenger, nullptr);
		vkDestroyDevice(device, nullptr);
		vkDestroyInstance(instance, nullptr);
		SDL_DestroyWindow(window);
	}
}

void VulkanEngine::draw()
{
	// wait until the GPU has finished rendering the last frame
	vkCheck(vkWaitForFences(device, 1, &renderFence, true, UINT64_MAX));
	vkCheck(vkResetFences(device, 1, &renderFence));

	// request image from the swapchain
	uint32_t swapchainImageIndex{};
	vkCheck(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, presentSemaphore,
	                              nullptr, &swapchainImageIndex));

	// now that we are sure that the commands finished executing,
	// we can safely reset the command buffer to begin recording again.
	vkCheck(vkResetCommandBuffer(mainCommandBuffer, 0));

	// naming it cmd for shorter writing
	auto cmd = mainCommandBuffer;

	// begin the command buffer recording. We will use this command buffer exactly once,
	// so we want to let Vulkan know that
	VkCommandBufferBeginInfo cmdBeginInfo{};
	cmdBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkCheck(vkBeginCommandBuffer(cmd, &cmdBeginInfo));

	const auto flash = abs(sin(frameNumber / 60.f));
	const VkClearValue clearValue =
		{{{0.0f, 0.0f, flash, 1.0f}}};

	// start the main renderpass
	// We will use the clear color from above and
	// the framebuffer of the index the swapchain gave us
	VkRenderPassBeginInfo rpInfo = {};
	rpInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
	rpInfo.renderPass = renderPass;
	rpInfo.renderArea.offset.x = 0;
	rpInfo.renderArea.offset.y = 0;
	rpInfo.renderArea.extent = windowExtent;
	rpInfo.framebuffer = framebuffers[swapchainImageIndex];
	rpInfo.clearValueCount = 1;
	rpInfo.pClearValues = &clearValue;

	vkCmdBeginRenderPass(cmd, &rpInfo, VK_SUBPASS_CONTENTS_INLINE);

	if (selectedShader == 0)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, trianglePipeline);
		vkCmdDraw(cmd, 3, 1, 0, 0);
	}
	else if (selectedShader == 1)
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, monochromeTrianglePipeline);
		vkCmdDraw(cmd, 3, 1, 0, 0);
	}
	else
	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipeline);
		VkDeviceSize offset = 0;
		vkCmdBindVertexBuffers(cmd, 0, 1, &triangleMesh.vertexBuffer.buffer, &offset);
		vkCmdDraw(cmd, triangleMesh.vertices.size(), 1, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkCheck(vkEndCommandBuffer(cmd));

	// we want to wait on the presentSemaphore, as that semaphore is signaled
	// when the swapchain is ready
	// we will signal the renderSemaphore, to signal that rendering has finished

	VkSubmitInfo submit{};
	submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

	constexpr VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

	submit.pWaitDstStageMask = &waitStage;

	submit.waitSemaphoreCount = 1;
	submit.pWaitSemaphores = &presentSemaphore;

	submit.signalSemaphoreCount = 1;
	submit.pSignalSemaphores = &renderSemaphore;

	submit.commandBufferCount = 1;
	submit.pCommandBuffers = &cmd;

	// submit command buffer to the queue and execute it.
	// renderFence will now block until the graphic commands finish execution
	vkCheck(vkQueueSubmit(graphicsQueue, 1, &submit, renderFence));

	// this will put the image we just rendered to into the visible window.
	// we want to wait on the renderSemaphore for that, 
	// as its necessary that drawing commands have finished before the image is displayed to the user
	VkPresentInfoKHR presentInfo{};
	presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

	presentInfo.swapchainCount = 1;
	presentInfo.pSwapchains = &swapchain;

	presentInfo.waitSemaphoreCount = 1;
	presentInfo.pWaitSemaphores = &renderSemaphore;

	presentInfo.pImageIndices = &swapchainImageIndex;

	vkCheck(vkQueuePresentKHR(graphicsQueue, &presentInfo));

	//increase the number of frames drawn
	frameNumber++;
}

void VulkanEngine::run()
{
	SDL_Event e{};
	auto bQuit = false;

	while (!bQuit)
	{
		while (SDL_PollEvent(&e) != 0)
		{
			if (e.type == SDL_QUIT)
			{
				bQuit = true;
			}
			else if (e.type == SDL_KEYDOWN)
			{
				if (e.key.keysym.sym == SDLK_SPACE)
				{
					selectedShader += 1;
					selectedShader %= 3;
				}
			}
		}

		draw();
	}
}
