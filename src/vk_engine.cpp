#include "vk_engine.h"

#pragma warning(push, 0)

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

#pragma warning(pop)

#include <iostream>
#include <vk_initializers.h>
#include <vk_types.h>
#include <VkBootstrap.h>

void vkCheck(const VkResult x)
{
	if (x != VK_SUCCESS)
	{
		std::cout << "Detected Vulkan error: " << x << '\n';
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
	debug_messenger = vkbInst.debug_messenger;

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
	chosenGPU = physicalDevice.physical_device;
}

void VulkanEngine::initSwapchain()
{
	vkb::SwapchainBuilder swapchainBuilder{chosenGPU, device, surface};

	vkb::Swapchain vkbSwapchain = swapchainBuilder
	                              .use_default_format_selection()
	                              .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
	                              .set_desired_extent(windowExtent.width, windowExtent.height)
	                              .build().value();

	swapchain = vkbSwapchain.swapchain;
	swapchainImages = vkbSwapchain.get_images().value();
	swapchainImageViews = vkbSwapchain.get_image_views().value();
	swapchainImageFormat = vkbSwapchain.image_format;
}

void VulkanEngine::cleanup() noexcept
{
	if (isInitialized)
	{
		vkDestroySwapchainKHR(device, swapchain, nullptr);
		for (auto swapchainImageView : swapchainImageViews)
		{
			vkDestroyImageView(device, swapchainImageView, nullptr);
		}

		vkDestroyDevice(device, nullptr);
		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkb::destroy_debug_utils_messenger(instance, debug_messenger, nullptr);
		vkDestroyInstance(instance, nullptr);
		SDL_DestroyWindow(window);
	}
}

void VulkanEngine::draw()
{
	// nothing yet
}

void VulkanEngine::run()
{
	SDL_Event e{};
	bool bQuit = false;

	// main loop
	while (!bQuit)
	{
		// Handle events on queue
		while (SDL_PollEvent(&e) != 0)
		{
			// close the window when user alt-f4s or clicks the X button
			if (e.type == SDL_QUIT)
				bQuit = true;
		}

		draw();
	}
}
