cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(sequence_demo)

set(CMAKE_BUILD_TYPE Release)

option(SPEED_TEST "SPEED TEST" OFF)
option(CONFORMANCE_TEST "conformance test" OFF)
option(ENCRYPT "encrypt" OFF)
if(SPEED_TEST)
    add_definitions(-DSPEED_TEST)
endif()
if(CONFORMANCE_TEST)
    add_definitions(-DCONFORMANCE_TEST)
endif()
if(ENCRYPT)
    add_definitions(-DENCRYPT)
endif()


# 设置可执行文件名
set(SEQUENCE_DEMO_SOURCES
    demo/main.cpp
    demo/utils/src/utils.cpp
)
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/build/build/generators/")
add_subdirectory("./sequence_inference")
add_executable(${PROJECT_NAME} ${SEQUENCE_DEMO_SOURCES})
include_directories("./demo/utils/include")
include_directories("./sequence_inference/include")
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)


# 链接 OpenCV 库和 ONNX Runtime 库
target_link_libraries(${PROJECT_NAME}
                      sequence_inference
                      opencv::opencv
                      onnxruntime::onnxruntime)

# 设置 C++ 标准
set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)
