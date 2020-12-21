// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vector>
#include <vk_types.h>

class VulkanEngine
{
public:
	bool isInitialized{false};
	int frameNumber{0};

	VkInstance instance{};
	VkDebugUtilsMessengerEXT debug_messenger{};
	VkPhysicalDevice chosenGPU{};
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

	VkPipelineLayout trianglePipelineLayout;
	VkPipeline trianglePipeline;

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
};


class PipelineBuilder
{
public:
	std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
	VkPipelineVertexInputStateCreateInfo vertexInputInfo;
	VkPipelineInputAssemblyStateCreateInfo inputAssembly;
	VkViewport viewport;
	VkRect2D scissor;
	VkPipelineRasterizationStateCreateInfo rasterizer;
	VkPipelineColorBlendAttachmentState colorBlendAttachment;
	VkPipelineMultisampleStateCreateInfo multisampling;
	VkPipelineLayout pipelineLayout;

	VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};
