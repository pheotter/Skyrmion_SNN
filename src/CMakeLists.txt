cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(leaky)

set(CMAKE_PREFIX_PATH /mnt/c/Users/romot/Downloads/libtorch)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} leaky.cpp sky.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${PYTHON_LIBRARIES}")


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)
