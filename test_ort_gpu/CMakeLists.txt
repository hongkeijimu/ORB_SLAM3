cmake_minimum_required(VERSION 3.16)
project(test_ort_gpu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(ONNXRUNTIME_ROOT "$ENV{HOME}/libs/onnxruntime-gpu/onnxruntime-linux-x64-gpu-1.18.0")

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

add_executable(test_ort_gpu test_ort_gpu.cpp)

target_link_libraries(test_ort_gpu
    ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
)