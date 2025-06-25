cmake_minimum_required(VERSION 3.10)
project(sequencer_inference_library)

set(SEQUENCE_INFERENCE_LIBRARY
    src/sequencer_inference.cpp
)
add_library(sequencer_inference ${SEQUENCE_INFERENCE_LIBRARY})

include_directories("./include")
find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${onnxruntime_INCLUDE_DIR})

set_target_properties(sequencer_inference PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES)  # 这个是必须的，否则会报错

set_target_properties(sequencer_inference PROPERTIES PUBLIC_HEADER "include/sequencer_inference.h")
install(TARGETS sequencer_inference)
