cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

project(ScannerComparison)

set(Scanner_DIR "$ENV{Scanner_DIR}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

set(OPENCV_COMPONENTS)
list(APPEND OPENCV_COMPONENTS core highgui imgproc video videoio)
list(APPEND OPENCV_COMPONENTS cudaarithm cudawarping cudaimgproc)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})

add_subdirectory(standalone)
add_subdirectory(peak)
add_subdirectory(ocv_decode)
add_subdirectory(kernel_sol)
add_subdirectory(readwrite)
# add_subdirectory(equal)
#add_subdirectory(caffe)
