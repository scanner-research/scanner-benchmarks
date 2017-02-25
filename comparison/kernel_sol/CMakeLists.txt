set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(CUDA_FOUND AND OpenCV_FOUND)
  add_executable(kernel_sol
    main.cpp
    $<TARGET_OBJECTS:util>)

  target_link_libraries(kernel_sol PRIVATE
    "${LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")

  foreach(TARGET ${HALIDE_TARGETS})
    halide_add_aot_library_dependency(kernel_sol ${TARGET})
  endforeach()
  target_link_libraries(kernel_sol PRIVATE
    scanner_halide)
endif()
