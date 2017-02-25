set(LIBRARIES
  "${OpenCV_LIBRARIES}"
  "${Boost_LIBRARIES}")

if(CUDA_FOUND AND OpenCV_FOUND)
  add_executable(peak_comparison
    main.cpp
    peak_video_decoder.cpp
    $<TARGET_OBJECTS:util>)

  target_link_libraries(peak_comparison PRIVATE
    "${LIBRARIES}")

  foreach(TARGET ${HALIDE_TARGETS})
    halide_add_aot_library_dependency(peak_comparison ${TARGET})
  endforeach()
  target_link_libraries(peak_comparison PRIVATE
    scanner_halide)
endif()
