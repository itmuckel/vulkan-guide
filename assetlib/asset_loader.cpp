#include "asset_loader.h"

#include <fstream>

bool assets::saveBinaryFile(const std::string& path, const AssetFile& file)
{
	std::ofstream outFile{};
	outFile.open(path, std::ios::binary | std::ios::out);

	outFile.write(file.type, 4);

	const auto version = file.version;
	outFile.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));

	const auto jsonLength = static_cast<uint32_t>(file.json.size());
	outFile.write(reinterpret_cast<const char*>(&jsonLength), sizeof(uint32_t));

	const auto blobLength = static_cast<uint32_t>(file.binaryBlob.size());
	outFile.write(reinterpret_cast<const char*>(&blobLength), sizeof(uint32_t));

	outFile.write(file.json.data(), jsonLength);
	outFile.write(file.binaryBlob.data(), file.binaryBlob.size());

	outFile.close();

	return true;
}

bool assets::loadBinaryFile(const std::string& path, AssetFile& outFile)
{
	std::ifstream inFile{};
	inFile.open(path, std::ios::binary);

	if (!inFile.is_open()) { return false; }

	// move file cursor to beginning
	inFile.seekg(0);

	inFile.read(outFile.type, 4);

	inFile.read(reinterpret_cast<char*>(&outFile.version), sizeof(uint32_t));

	uint32_t jsonLength{};
	inFile.read(reinterpret_cast<char*>(&jsonLength), sizeof(uint32_t));

	uint32_t blobLength{};
	inFile.read(reinterpret_cast<char*>(&blobLength), sizeof(uint32_t));

	outFile.json.resize(jsonLength);
	inFile.read(outFile.json.data(), jsonLength);

	outFile.binaryBlob.resize(blobLength);
	inFile.read(outFile.binaryBlob.data(), blobLength);

	return true;
}

assets::CompressionMode assets::parseCompression(const std::string& value)
{
	if (value == "LZ4")
	{
		return CompressionMode::LZ4;
	}

	return CompressionMode::None;
}
