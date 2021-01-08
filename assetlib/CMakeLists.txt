cmake_minimum_required(VERSION 3.8)

add_library(assetlib STATIC 
		"asset_loader.h"
		"asset_loader.cpp"
		"texture_asset.h"
		"texture_asset.cpp" "mesh_asset.cpp")

target_include_directories(assetlib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_include_directories(assetlib PUBLIC
        "C:/src/vcpkg/installed/x64-windows/include")
target_link_directories(assetlib PUBLIC
        "C:/src/vcpkg/installed/x64-windows/lib")

target_link_libraries(assetlib PRIVATE nlohmann_json lz4)