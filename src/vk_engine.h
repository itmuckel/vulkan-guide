// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <deque>
#include <unordered_map>
#include <functional>
#include <vector>
#include <vk_types.h>
#include <glm/glm.hpp>


#include "vk_descriptors.h"
#include "vk_mesh.h"

struct Material
{
	VkDescriptorSet textureSet;
	VkPipeline pipeline{};
	VkPipelineLayout pipelineLayout{};
};

struct Texture
{
	AllocatedImage image{};
	VkImageView imageView{};
};

struct RenderObject
{
	Mesh* mesh{};
	Material* material{};
	glm::mat4 transformMatrix{};
};

struct FrameData
{
	VkSemaphore presentSemaphore{};
	VkSemaphore renderSemaphore{};
	VkFence renderFence{};

	VkCommandPool commandPool{};
	VkCommandBuffer mainCommandBuffer{};

	VkDescriptorSet globalDescriptor{};
	AllocatedBuffer cameraBuffer{};

	VkDescriptorSet objectDescriptor{};
	AllocatedBuffer objectBuffer{};
};

struct GpuCameraData
{
	glm::mat4 view{};
	glm::mat4 projection{};
	glm::mat4 viewproj{};
};

struct GpuSceneData
{
	glm::vec4 fogColor{}; // w is for exponent
	glm::vec4 fogDistances{}; // x for min, y for max, zw unused
	glm::vec4 ambientColor{};
	glm::vec4 sunlightDirection{}; // w for sun power
	glm::vec4 sunlightColor{};
};

struct GpuObjectData
{
	glm::mat4 modelMatrix{};
};

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors{};

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

struct Camera
{
	float height{0.f};

	glm::vec3 pos{0.f, height, 3.f};
	glm::vec3 up{0.f, 1.f, 0.f};
	glm::vec3 direction{0.f, 0.f, -1.f};

	float pitch = 0.f;
	float yaw = -90.f;

	static const float MOVE_SPEED;
	static const float X_SPEED;
	static const float Y_SPEED;

	void calculateDirection(float deltaYaw, float deltaPitch);
};

struct MeshPushConstants
{
	glm::vec4 data{};
	glm::mat4 renderMatrix{};
};

struct UploadContext
{
	VkFence uploadFence{};
	VkCommandPool commandPool{};
};

constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine
{
public:
	bool isInitialized{false};
	DeletionQueue mainDeletionQueue{};

	VkInstance instance{};
	VkDebugUtilsMessengerEXT debugMessenger{};
	VkPhysicalDevice chosenGpu{};
	VkPhysicalDeviceProperties gpuProperties{};
	VkDevice device{};
	VkSurfaceKHR surface{};

	VkExtent2D windowExtent{static_cast<uint32_t>(800), static_cast<uint32_t>(600)};

	struct SDL_Window* window{nullptr};

	// -------- swapchain

	VkSwapchainKHR swapchain{};

	std::vector<VkImage> swapchainImages{};
	std::vector<VkImageView> swapchainImageViews{};
	VkFormat swapchainImageFormat{};

	VkImageView depthImageView{};
	AllocatedImage depthImage{};
	VkFormat depthFormat{};

	FrameData frames[FRAME_OVERLAP];

	// -------- commands

	VkQueue graphicsQueue{};
	uint32_t graphicsQueueFamily{};

	// --------- render passes

	VkRenderPass renderPass;
	std::vector<VkFramebuffer> framebuffers;


	// --------- memory

	VmaAllocator allocator{};

	// --------- descriptors

	VkDescriptorPool descriptorPool{};
	DescriptorLayoutCache descriptorLayoutCache{};
	DescriptorAllocator descriptorAllocator{};

	VkDescriptorSetLayout globalSetLayout{};
	VkDescriptorSetLayout objectSetLayout{};
	VkDescriptorSetLayout singleTextureSetLayout{};

	// --------- control flow

	Camera camera{};
	GpuSceneData sceneParameters{};
	AllocatedBuffer sceneParametersBuffer{};

	int frameNumber{0};
	int selectedShader{0};

	std::vector<RenderObject> renderables;
	std::unordered_map<std::string, Material> materials;
	std::unordered_map<std::string, Mesh> meshes;
	std::unordered_map<std::string, Texture> loadedTextures;

	UploadContext uploadContext{};

	Material* createMaterial(VkPipeline pipeline, VkPipelineLayout layout, const std::string& name);
	Material* getMaterial(const std::string& name);

	Mesh* getMesh(const std::string& name);

	void loadImages();

	[[nodiscard]]
	AllocatedBuffer createBuffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage) const;

	void initDescriptors();

	void immediateSubmit(std::function<void(VkCommandBuffer vmd)>&& function) const;
	void uploadMesh(Mesh& mesh);

	// initializes everything in the engine
	void init();

	void initScene();

	// shuts down the engine
	void cleanup();

	// draw loop
	void draw();

	[[nodiscard]]
	const FrameData& getCurrentFrame() const;

	void drawObjects(VkCommandBuffer cmd, RenderObject* first, int count);

	// run main loop
	void run();

private:
	void initVulkan();

	[[nodiscard]]
	size_t padUniformBufferSize(size_t originalSize) const;

	void initSwapchain();

	void initCommands();

	void initDefaultRenderpass();

	void initFramebuffers();

	void initSyncStructures();

	/// loads a shader module from a spir-v file. Returns false if it errors
	bool loadShaderModule(const char* filePath, VkShaderModule& outShaderModule) const;

	void initPipelines();

	void loadMeshes();
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
	VkPipelineDepthStencilStateCreateInfo depthStencil{};
	VkPipelineMultisampleStateCreateInfo multisampling{};

	VkPipelineLayout pipelineLayout{};

	VkPipeline buildPipeline(VkDevice device, VkRenderPass pass);
};
