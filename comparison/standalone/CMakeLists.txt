add_executable(standalone_comparison
  main.cpp
  $<TARGET_OBJECTS:util>)

target_link_libraries(standalone_comparison
  ${LIBRARIES})
