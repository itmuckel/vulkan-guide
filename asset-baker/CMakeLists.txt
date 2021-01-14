set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "/MD")

link_directories("../third_party/nvidia-texture-tools/project/vc2017/Release.x64/bin")
include_directories("../third_party/nvidia-texture-tools/src/nvtt")

# Add source to this project's executable.
add_executable (baker
    "asset_main.cpp")
#set_property(TARGET baker PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:extra>")

target_include_directories(baker PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

message("${CMAKE_CURRENT_DIRECTORY}")

target_link_libraries(baker PUBLIC assetlib stb_image tinyobjloader nvtt lz4 glm assimp-vc142-mt)

