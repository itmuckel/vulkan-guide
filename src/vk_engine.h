// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <deque>
#include <functional>
#include <vector>
#include <vk_types.h>
#include <glm/glm.hpp>

#include "vk_mesh.h"

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void pushFunction(std::function<void()>&& function)
	{
		deletors.push_back(function);
	}

	void flush()
	{
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it += 1)
		{
			(*it)(); //call functors
		}

		deletors.clear();
	}
};

struct MeshPushConstants
{
	glm::vec4 data{};
	glm::mat4 renderMatrix{};
};

class VulkanEngine
{
public:
	bool isInitialized{false};
	DeletionQueue mainDeletionQueue{};

	VkInstance instance{};
	VkDebugUtilsMessengerEXT debugMessenger{};
	VkPhysicalDevice chosenGpu{};
	VkDevice device{};
	VkSurfaceKHR surface{};

	VkExtent2D windowExtent{static_cast<uint32_t>(800), static_cast<uint32_t>(600)};

	struct SDL_Window* window{nullptr};

	// -------- swapchain

	VkSwapchainKHR swapchain{};

	VkFormat swapchainImageFormat{};

	std::vector<VkImage> swapchainImages{};

	std::vector<VkImageView> swapchainImageViews{};

	// -------- commands

	VkQueue graphicsQueue{};
	uint32_t graphicsQueueFamily{};

	VkCommandPool commandPool{};
	VkCommandBuffer mainCommandBuffer{};

	// --------- render passes

	VkRenderPass renderPass;
	std::vector<VkFramebuffer> framebuffers;

	// --------- synchronization

	VkSemaphore presentSemaphore{};
	VkSemaphore renderSemaphore{};
	VkFence renderFence{};

	// --------- pipeline

	VkPipelineLayout monochromeTrianglePipelineLayout;
	VkPipelineLayout meshPipelineLayout{};
	VkPipeline monochromeTrianglePipeline;
	VkPipeline trianglePipeline;
	VkPipeline meshPipeline;
	Mesh triangleMesh;

	// --------- memory

	VmaAllocator allocator{};

	// --------- control flow

	int frameNumber{0};
	int selectedShader{0};

	// initializes everything in the engine
	void init();

	// shuts down the engine
	void cleanup();

	// draw loop
	void draw();

	// run main loop
	void run();

private:
	void initVulkan();

	void initSwapchain();

	void initCommands();

	void initDefaultRenderpass();

	void initFramebuffers();

	void initSyncStructures();

	/// loads a shader module from a spir-v file. Returns false if it errors
	bool loadShaderModule(const char* filePath, VkShaderModule& outShaderModule);

	void initPipelines();

	void loadMeshes();

	void uploadMesh(Mesh& mesh);
};

class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages{};
	VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
	VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
	VkViewport viewport{};
	VkRect2D scissor{};
	VkPipelineRasterizationStateCreateInfo rasterizer{};
	VkPipelineColorBlendAttachmentState colorBlendAttachment{};
	VkPipelineMultisampleStateCreateInfo multisampling{};

	VkPipelineLayout pipelineLayout{};

	VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};
