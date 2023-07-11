cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(leaky)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_PREFIX_PATH /mnt/c/Users/romot/Downloads/libtorch)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Torch_INCLUDE_DIRS})

add_library(${PROJECT_NAME} leaky.cpp IEEE754.cpp sky.cpp)
target_include_directories(${PROJECT_NAME} PUBLIC include)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${PYTHON_LIBRARIES}")


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

enable_testing()

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

add_executable(leaky_test ../test/ut_leaky.cpp)
target_link_libraries(leaky_test
  PRIVATE
    GTest::GTest
    ${PROJECT_NAME})

include(GoogleTest)
gtest_discover_tests(leaky_test)
