#include "vk_descriptors.h"

#include <algorithm>
#include <iostream>

void DescriptorAllocator::resetPools()
{
	for (const auto& p : usedPools)
	{
		vkResetDescriptorPool(device, p, 0);
	}

	// TODO: Isn't this a memory leak?
	if (!freePools.empty())
	{
		std::cerr << "possible memory leak in DescriptorAllocator::resetPools" << '\n';
	}

	freePools = usedPools;
	usedPools.clear();
	currentPool = nullptr;
}

bool DescriptorAllocator::allocate(VkDescriptorSet* set, VkDescriptorSetLayout layout)
{
	if (currentPool == nullptr)
	{
		currentPool = grabPool();
		usedPools.push_back(currentPool);
	}

	VkDescriptorSetAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocInfo.pSetLayouts = &layout;
	allocInfo.descriptorPool = currentPool;
	allocInfo.descriptorSetCount = 1;

	auto allocResult = vkAllocateDescriptorSets(device, &allocInfo, set);

	switch (allocResult)
	{
	case VK_SUCCESS:
		return true;
	case VK_ERROR_FRAGMENTED_POOL:
	case VK_ERROR_OUT_OF_POOL_MEMORY:
		// need to reallocate!
		break;
	default:
		// unrecoverable error
		return false;
	}

	currentPool = grabPool();
	usedPools.push_back(currentPool);

	allocResult = vkAllocateDescriptorSets(device, &allocInfo, set);

	if (allocResult == VK_SUCCESS)
	{
		return true;
	}

	// even the reallocation failed
	return false;
}

void DescriptorAllocator::init(const VkDevice newDevice)
{
	device = newDevice;
}

void DescriptorAllocator::cleanup()
{
	for (const auto& p : freePools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
	for (const auto& p : usedPools)
	{
		vkDestroyDescriptorPool(device, p, nullptr);
	}
}

VkDescriptorPool createPool(const VkDevice device, const DescriptorAllocator::PoolSizes& poolSizes,
                            int count, VkDescriptorPoolCreateFlags flags)
{
	std::vector<VkDescriptorPoolSize> sizes{};
	sizes.reserve(poolSizes.sizes.size());
	for (const auto& [type, descriptorCount] : poolSizes.sizes)
	{
		sizes.push_back({type, static_cast<uint32_t>(descriptorCount * count)});
	}
	VkDescriptorPoolCreateInfo poolInfo{};
	poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	poolInfo.flags = flags;
	poolInfo.maxSets = count;
	poolInfo.poolSizeCount = static_cast<uint32_t>(sizes.size());
	poolInfo.pPoolSizes = sizes.data();

	VkDescriptorPool descriptorPool{};
	vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool);

	return descriptorPool;
}

VkDescriptorPool DescriptorAllocator::grabPool()
{
	if (!freePools.empty())
	{
		const auto pool = freePools.back();
		freePools.pop_back();

		return pool;
	}

	return createPool(device, descriptorSizes, 1000, 0);
}

bool DescriptorLayoutCache::DescriptorLayoutInfo::operator==(const DescriptorLayoutInfo& other) const
{
	if (other.bindings.size() != bindings.size())
	{
		return false;
	}

	for (size_t i = 0; i < bindings.size(); i += 1)
	{
		if (other.bindings[i].binding != bindings[i].binding)
		{
			return false;
		}
		if (other.bindings[i].descriptorType != bindings[i].descriptorType)
		{
			return false;
		}
		if (other.bindings[i].descriptorCount != bindings[i].descriptorCount)
		{
			return false;
		}
		if (other.bindings[i].stageFlags != bindings[i].stageFlags)
		{
			return false;
		}
	}

	return true;
}

size_t DescriptorLayoutCache::DescriptorLayoutInfo::hash() const
{
	size_t result = std::hash<size_t>()(bindings.size());

	for (const auto& b : bindings)
	{
		//pack the binding data into a single int64. Not fully correct but its ok
		size_t bindingHash = b.binding | b.descriptorType << 8 | b.descriptorCount << 16 | b.stageFlags << 24;

		//shuffle the packed binding data and xor it with the main hash
		result ^= std::hash<size_t>()(bindingHash);
	}

	return result;
}

void DescriptorLayoutCache::init(VkDevice newDevice)
{
	device = newDevice;
}

void DescriptorLayoutCache::cleanup()
{
	for (const auto& [_, descriptorSetLayout] : layoutCache)
	{
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
	}
}

VkDescriptorSetLayout DescriptorLayoutCache::createDescriptorLayout(VkDescriptorSetLayoutCreateInfo* info)
{
	DescriptorLayoutInfo layoutInfo{};
	layoutInfo.bindings.reserve(info->bindingCount);
	auto isSorted = true;
	uint32_t lastBinding = -1;

	for (size_t i = 0; i < info->bindingCount; i += 1)
	{
		layoutInfo.bindings.push_back(info->pBindings[i]);

		if (info->pBindings[i].binding > lastBinding)
		{
			lastBinding = info->pBindings[i].binding;
		}
		else
		{
			isSorted = false;
		}
	}

	if (!isSorted)
	{
		std::sort(layoutInfo.bindings.begin(), layoutInfo.bindings.end(), [](auto& a, auto& b)
		{
			return a.binding < b.binding;
		});
	}

	const auto it = layoutCache.find(layoutInfo);
	if (it != layoutCache.end())
	{
		return it->second;
	}

	// didn't exist in cache, create new one
	VkDescriptorSetLayout layout{};
	vkCreateDescriptorSetLayout(device, info, nullptr, &layout);

	layoutCache[layoutInfo] = layout;

	return layout;
}

DescriptorBuilder DescriptorBuilder::begin(DescriptorLayoutCache* layoutCache, DescriptorAllocator* allocator)
{
	DescriptorBuilder builder{};
	builder.cache = layoutCache;
	builder.alloc = allocator;

	return builder;
}

DescriptorBuilder& DescriptorBuilder::bindBuffer(uint32_t binding, VkDescriptorBufferInfo* bufferInfo,
                                                 VkDescriptorType type, VkShaderStageFlags stageFlags)
{
	VkDescriptorSetLayoutBinding newBinding{};

	newBinding.descriptorCount = 1;
	newBinding.descriptorType = type;
	newBinding.pImmutableSamplers = nullptr;
	newBinding.stageFlags = stageFlags;
	newBinding.binding = binding;

	bindings.push_back(newBinding);

	VkWriteDescriptorSet newWrite{};
	newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

	newWrite.descriptorCount = 1;
	newWrite.descriptorType = type;
	newWrite.pBufferInfo = bufferInfo;
	newWrite.dstBinding = binding;

	writes.push_back(newWrite);

	return *this;
}

DescriptorBuilder& DescriptorBuilder::bindImage(uint32_t binding, VkDescriptorImageInfo* imageInfo,
                                                VkDescriptorType type, VkShaderStageFlags stageFlags)
{
	VkDescriptorSetLayoutBinding newBinding{};

	newBinding.descriptorCount = 1;
	newBinding.descriptorType = type;
	newBinding.pImmutableSamplers = nullptr;
	newBinding.stageFlags = stageFlags;
	newBinding.binding = binding;

	bindings.push_back(newBinding);

	VkWriteDescriptorSet newWrite{};
	newWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;

	newWrite.descriptorCount = 1;
	newWrite.descriptorType = type;
	newWrite.pImageInfo = imageInfo;
	newWrite.dstBinding = binding;

	writes.push_back(newWrite);

	return *this;
}

bool DescriptorBuilder::build(VkDescriptorSet& set, VkDescriptorSetLayout& layout)
{
	VkDescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	layoutInfo.pBindings = bindings.data();
	layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
	layout = cache->createDescriptorLayout(&layoutInfo);

	if (!alloc->allocate(&set, layout))
	{
		return false;
	}

	for (auto& w : writes)
	{
		w.dstSet = set;
	}

	vkUpdateDescriptorSets(alloc->device, writes.size(), writes.data(), 0, nullptr);

	return true;
}

bool DescriptorBuilder::build(VkDescriptorSet& set)
{
	VkDescriptorSetLayout layout{};

	return build(set, layout);
}
