#pragma once
#include "asset_loader.h"

namespace assets {

	enum class TextureFormat : uint32_t
	{
		Unknown = 0,
		RGBA8
	};

	struct PageInfo {
		uint32_t width{};
		uint32_t height{};
		uint32_t compressedSize{};
		uint32_t originalSize{};
	};

	struct TextureInfo {
		uint64_t textureSize{};
		TextureFormat textureFormat{};
		CompressionMode compressionMode{};

		std::string originalFile{};
		std::vector<PageInfo> pages{};
	};

	TextureInfo readTextureInfo(AssetFile* file);

	void unpackTexture(TextureInfo* info, const char* sourcebuffer, size_t sourceSize, char* destination);

	void unpackTexturePage(TextureInfo* info, int pageIndex, char* sourcebuffer, char* destination);

	AssetFile packTexture(TextureInfo* info, void* pixelData);
}