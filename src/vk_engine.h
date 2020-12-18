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

	// -------- swapchain

	VkSwapchainKHR swapchain{};

	VkFormat swapchainImageFormat{};

	std::vector<VkImage> swapchainImages{};

	std::vector<VkImageView> swapchainImageViews{};

	struct SDL_Window* window{nullptr};

	// initializes everything in the engine
	void init();

	void initSwapchain();

	// shuts down the engine
	void cleanup() noexcept;

	// draw loop
	void draw();

	// run main loop
	void run();

private:
	void initVulkan();
};
