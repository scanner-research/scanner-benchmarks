set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

add_executable(readwrite_test
  main.cpp
  ${PROTO_SRCS})

target_include_directories(readwrite_test PUBLIC
  "${Scanner_DIR}")

target_link_libraries(readwrite_test
  "${LIBRARIES}"
  "${PUBLIC_LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}"
  "${Scanner_DIR}/build/libscanner.so")
