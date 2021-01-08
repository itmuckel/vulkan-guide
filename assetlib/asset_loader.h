#pragma once

#include <string>
#include <vector>

namespace assets
{
	struct AssetFile
	{
		char type[4];
		uint32_t version{};
		std::string json;
		std::vector<char> binaryBlob{};
	};

	enum class CompressionMode: uint32_t
	{
		None,
		LZ4
	};

	bool saveBinaryFile(const std::string& path, const AssetFile& file);

	bool loadBinaryFile(const std::string& path, AssetFile& outFile);

	CompressionMode parseCompression(const std::string& value);
}
