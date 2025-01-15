cmake_minimum_required(VERSION 3.10)

# 项目名称
project(Datagenerator)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ------------------------------------------------------------------------------
# 添加头文件路径
# ------------------------------------------------------------------------------
include_directories(${PROJECT_SOURCE_DIR})

# ------------------------------------------------------------------------------
# 查找需要的依赖库（PCL、OpenCV、Eigen、OctoMap）
# ------------------------------------------------------------------------------
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

# ------------------------------------------------------------------------------
# 如果你想自动查找 Gurobi，可以写一个 FindGurobi.cmake，
# 否则这里示范手动指定 Gurobi 的路径（硬编码）
# ------------------------------------------------------------------------------
set(GUROBI_HOME "/usr/local/gurobi1200/linux64")  # 如果你安装在别处，请修改
include_directories(${GUROBI_HOME}/include)       # 等效于 -I/usr/local/gurobi1200/linux64/include
link_directories(${GUROBI_HOME}/lib)             # 等效于 -L/usr/local/gurobi1200/linux64/lib

# ------------------------------------------------------------------------------
# 添加可执行文件
# ------------------------------------------------------------------------------
add_executable(Datagenerator main.cpp)

# ------------------------------------------------------------------------------
# 链接依赖库
# ------------------------------------------------------------------------------
target_link_libraries(Datagenerator
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${OCTOMAP_LIBRARIES}

    # ------------------- Gurobi 库 -------------------
    # 如果你使用动态库的方式（推荐）
    gurobi_c++         # 等效于 -lgurobi_c++
    -l:libgurobi.so.12.0.0
    #
    # 如果需要 pthread 或其他库，也可以在这里添加:
    # pthread
)

# ------------------------------------------------------------------------------
# 设置编译器选项
# ------------------------------------------------------------------------------
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()
