add_executable(readwrite_test
  main.cpp)

target_link_libraries(readwrite_test
  "${LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}")
