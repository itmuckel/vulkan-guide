#include "texture_asset.h"
#include <nlohmann/json.hpp>
#include <lz4.h>

using json = nlohmann::json;

assets::TextureFormat parseFormat(const std::string& value)
{
	if (value == "RGBA8")
	{
		return assets::TextureFormat::RGBA8;
	}

	return assets::TextureFormat::Unknown;
}

assets::TextureInfo assets::readTextureInfo(AssetFile* file)
{
	TextureInfo info;

	auto textureMetadata = json::parse(file->json);

	const std::string formatString = textureMetadata["format"];
	info.textureFormat = parseFormat(formatString);

	const std::string compressionString = textureMetadata["compression"];
	info.compressionMode = parseCompression(compressionString);

	info.textureSize = textureMetadata["buffer_size"];
	info.originalFile = textureMetadata["original_file"];

	for (const auto& [key, value] : textureMetadata["pages"].items())
	{
		PageInfo page;

		page.compressedSize = value["compressed_size"];
		page.originalSize = value["original_size"];
		page.width = value["width"];
		page.height = value["height"];

		info.pages.push_back(page);
	}


	return info;
}

void assets::unpackTexture(TextureInfo* info, const char* sourcebuffer, size_t sourceSize, char* destination)
{
	if (info->compressionMode == CompressionMode::LZ4)
	{
		for (auto& page : info->pages)
		{
			LZ4_decompress_safe(sourcebuffer, destination, page.compressedSize, page.originalSize);
			sourcebuffer += page.compressedSize;
			destination += page.originalSize;
		}
	}
	else
	{
		memcpy(destination, sourcebuffer, sourceSize);
	}
}

void assets::unpackTexturePage(TextureInfo* info, int pageIndex, char* sourcebuffer, char* destination)
{
	auto* source = sourcebuffer;
	for (auto i = 0; i < pageIndex; i++)
	{
		source += info->pages[i].compressedSize;
	}

	if (info->compressionMode == CompressionMode::LZ4)
	{
		// size doesn't fully match -> it's compressed
		if (info->pages[pageIndex].compressedSize != info->pages[pageIndex].originalSize)
		{
			LZ4_decompress_safe(source, destination, info->pages[pageIndex].compressedSize,
			                    info->pages[pageIndex].originalSize);
		}
		else
		{
			// size matched -> uncompressed page
			memcpy(destination, source, info->pages[pageIndex].originalSize);
		}
	}
	else
	{
		memcpy(destination, source, info->pages[pageIndex].originalSize);
	}
}


assets::AssetFile assets::packTexture(TextureInfo* info, void* pixelData)
{
	//core file header
	AssetFile file;
	file.type[0] = 'T';
	file.type[1] = 'E';
	file.type[2] = 'X';
	file.type[3] = 'I';
	file.version = 1;


	auto* pixels = static_cast<char*>(pixelData);
	std::vector<char> page_buffer;
	for (auto& p : info->pages)
	{
		page_buffer.resize(p.originalSize);


		//compress buffer into blob
		const auto compressStaging = LZ4_compressBound(p.originalSize);

		page_buffer.resize(compressStaging);

		auto compressedSize = LZ4_compress_default(pixels, page_buffer.data(), p.originalSize, compressStaging);


		const auto compressionRate =
			static_cast<float>(compressedSize) /
			static_cast<float>(info->textureSize);

		// if the compression is more than 80% of the original size, it's not worth to use it
		if (compressionRate > 0.8f)
		{
			compressedSize = p.originalSize;
			page_buffer.resize(compressedSize);

			memcpy(page_buffer.data(), pixels, compressedSize);
		}
		else
		{
			page_buffer.resize(compressedSize);
		}
		p.compressedSize = compressedSize;

		file.binaryBlob.insert(file.binaryBlob.end(), page_buffer.begin(), page_buffer.end());

		//advance pixel pointer to next page
		pixels += p.originalSize;
	}
	json textureMetadata;
	textureMetadata["format"] = "RGBA8";

	textureMetadata["buffer_size"] = info->textureSize;
	textureMetadata["original_file"] = info->originalFile;
	textureMetadata["compression"] = "LZ4";

	std::vector<json> pageJson{};
	for (auto& p : info->pages)
	{
		json page;
		page["compressed_size"] = p.compressedSize;
		page["original_size"] = p.originalSize;
		page["width"] = p.width;
		page["height"] = p.height;
		pageJson.push_back(page);
	}
	textureMetadata["pages"] = pageJson;

	const auto stringified = textureMetadata.dump();
	file.json = stringified;

	return file;
}
