
# Add source to this project's executable.
add_executable(vulkan_guide
        "main.cpp"
        "vk_engine.cpp"
        "vk_engine.h"
        "vk_types.h"
        "vk_initializers.cpp"
        "vk_initializers.h"
        "vk_mesh.cpp"
        "vk_mesh.h")


set_property(TARGET vulkan_guide PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "$<TARGET_FILE_DIR:vulkan_guide>")

target_include_directories(vulkan_guide PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(vulkan_guide PUBLIC
        "C:/src/vcpkg/installed/x64-windows/include")
target_link_directories(vulkan_guide PUBLIC
        "C:/src/vcpkg/installed/x64-windows/lib")
target_link_directories(vulkan_guide PUBLIC
        "C:/src/vcpkg/installed/x64-windows/lib/manual-link")

target_link_libraries(vulkan_guide vkbootstrap vma glm tinyobjloader imgui stb_image)

target_link_libraries(vulkan_guide Vulkan::Vulkan sdl2)

add_dependencies(vulkan_guide Shaders)
