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
find_package(Boost COMPONENTS thread program_options regex python system REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS})
find_package(Caffe REQUIRED)
find_package(TinyToml REQUIRED)
find_package(GFlags REQUIRED)
find_package(Glog REQUIRED)
find_package(FFmpeg REQUIRED)
find_package(Storehouse REQUIRED CONFIG
  PATHS "${SCANNER_ROOT_DIR}/thirdparty/build/bin/storehouse")

find_package(Halide REQUIRED)
include(${CMAKE_SOURCE_DIR}/cmake/Util/HalideGenerator.cmake)
include_directories("${HALIDE_INCLUDE_DIR}")
include_directories("${PROJECT_BINARY_DIR}/generator_genfiles")


find_package(CUDA REQUIRED)

add_definitions(-DUSE_OPENCV)
add_definitions(-DHAVE_CUDA)

include_directories(
  "."
  ${SCANNER_ROOT_DIR}
  ${SCANNER_ROOT_DIR}/build
  ${SCANNER_ROOT_DIR}/thirdparty/storehouse/
  ${FFMPEG_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CAFFE_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${TINYTOML_INCLUDE_DIR})

set(LIBRARIES
  ${SCANNER_ROOT_DIR}/build/libscanner.so
  ${FFMPEG_LIBRARIES}
  "/opt/ffmpeg-3.2.2/lib/libswscale.so"
  ${HALIDE_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CAFFE_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${PROTOBUF_LIBRARY}
  ${GFLAGS_LIBRARIES}
  ${GLOG_LIBRARIES}
  ${STOREHOUSE_LIBRARIES}
  "-ltinfo"
  )

set(HALIDE_TARGETS)
macro(add_halide_target SRC TARGET)
  if (NOT HALIDE_FOUND)
    find_package(Halide REQUIRED)
    include(${CMAKE_SOURCE_DIR}/cmake/Util/HalideGenerator.cmake)
    include_directories("${HALIDE_INCLUDE_DIR}")
    include_directories("${PROJECT_BINARY_DIR}/generator_genfiles")
  endif()

  get_filename_component(NAME ${SRC} NAME_WE)
  set(GENERATOR ${NAME}.generator)
  halide_add_generator(${GENERATOR}
    SRCS ${SRC})
  halide_add_aot_library(${NAME}
    GENERATOR_TARGET ${GENERATOR}
    GENERATOR_ARGS target=${TARGET})
  list(APPEND HALIDE_TARGETS ${NAME})
  set(HALIDE_TARGETS ${HALIDE_TARGETS} PARENT_SCOPE)
endmacro()

add_subdirectory(util)
add_subdirectory(standalone)
add_subdirectory(peak)
add_subdirectory(ocv_decode)
#add_subdirectory(kernel_sol)
#add_subdirectory(readwrite)
# add_subdirectory(equal)
#add_subdirectory(caffe)
