cmake_minimum_required(VERSION 3.16)
project(pedestrian_detection)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    /usr/local/cuda/include
    /usr/src/tensorrt/include
)

link_directories(
    /usr/local/cuda/lib64
    /usr/lib/aarch64-linux-gnu
)

add_executable(pedestrian_test
    app/main.cpp
    src/pedestrian/PedestrianDetector.cpp
)

target_link_libraries(pedestrian_test
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)

# 🔥 REAL FIX
target_link_options(pedestrian_test PRIVATE -Wl,--unresolved-symbols=ignore-in-shared-libs)
