cmake_minimum_required(VERSION 3.10)

# 项目名称
project(Datagenerator)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 添加头文件路径
include_directories(${PROJECT_SOURCE_DIR})

# 查找需要的依赖库
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

# 添加可执行文件
add_executable(Datagenerator main.cpp)

# 链接依赖库
target_link_libraries(Datagenerator
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OCTOMAP_LIBRARIES}
)

# 设置编译器选项
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
