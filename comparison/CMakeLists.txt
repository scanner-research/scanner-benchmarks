cmake_minimum_required(VERSION 3.2.0 FATAL_ERROR)

project(ScannerComparison)

set(SCANNER_ROOT_DIR "" CACHE PATH "")
if (NOT "$ENV{Scanner_DIR}" STREQUAL "")
  set(SCANNER_ROOT_DIR $ENV{Scanner_DIR} CACHE PATH " " FORCE)
endif()
if ("${SCANNER_ROOT_DIR}" STREQUAL "")
  message(FATAL_ERROR "Must provide a valid SCANNER_ROOT_DIR")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

set(OPENCV_COMPONENTS)
list(APPEND OPENCV_COMPONENTS core highgui imgproc video videoio)
list(APPEND OPENCV_COMPONENTS cudaarithm cudawarping cudaimgproc cudacodec
  cudaoptflow)

find_package(SaneProtobuf REQUIRED)
find_package(Boost COMPONENTS thread program_options regex python REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})
find_package(Caffe REQUIRED)
find_package(TinyToml REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)

find_package(CUDA REQUIRED)

add_definitions(-DUSE_OPENCV)
add_definitions(-DHAVE_CUDA)

include_directories(
  "."
  ${BOOST_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CAFFE_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${TINYTOML_INCLUDE_DIR})

set(LIBRARIES
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CAFFE_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${PROTOBUF_LIBRARY}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES})

add_subdirectory(util)
add_subdirectory(standalone)
#add_subdirectory(peak)
add_subdirectory(ocv_decode)
#add_subdirectory(kernel_sol)
#add_subdirectory(readwrite)
# add_subdirectory(equal)
#add_subdirectory(caffe)
