cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(mesh_sampling)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找PCL库并指定所需组件
find_package(PCL 1.9 REQUIRED COMPONENTS common io filters visualization)

# 包含PCL头文件
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# 添加可执行文件
add_executable(mesh_sampling mesh_sampling.cpp)

# 链接所需的PCL库
target_link_libraries(mesh_sampling ${PCL_LIBRARIES})
