add_executable(ocv_decode
  main.cpp
  $<TARGET_OBJECTS:util>)

target_link_libraries(ocv_decode
  "${LIBRARIES}"
  "${PLATFORM_LINK_FLAGS}")
