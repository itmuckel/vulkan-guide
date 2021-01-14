cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "/MD")

add_library (assetlib STATIC 
"mesh_asset.h"
"mesh_asset.cpp"
"texture_asset.h"
"texture_asset.cpp"
"material_asset.h"
"material_asset.cpp"
"prefab_asset.h"
"prefab_asset.cpp"
"asset_loader.h"
"asset_loader.cpp"
)

target_include_directories(assetlib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(assetlib PRIVATE lz4)