cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

project(ScannerComparison)

set(Scanner_DIR "$ENV{Scanner_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

add_subdirectory(standalone)
add_subdirectory(peak)
add_subdirectory(ocv_decode)
add_subdirectory(kernel_sol)
add_subdirectory(readwrite)
# add_subdirectory(equal)
#add_subdirectory(caffe)
