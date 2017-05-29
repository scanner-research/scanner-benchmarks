if(CUDA_FOUND AND OpenCV_FOUND)
  add_halide_target(${SCANNER_ROOT_DIR}/stdlib/caffe/caffe_input_transformer_cpu.cpp host)
  if (CUDA_FOUND)
    add_halide_target(${SCANNER_ROOT_DIR}/stdlib/caffe/caffe_input_transformer_gpu.cpp cuda)
  endif()

  add_executable(peak_comparison
    main.cpp
    peak_cpu_video_decoder.cpp
    peak_gpu_video_decoder.cpp
    $<TARGET_OBJECTS:util>)

  target_link_libraries(peak_comparison PRIVATE
    "-pthread"
    ${LIBRARIES}
    "-lcudart"
    "-lcuda"
    "-lnvcuvid")

  foreach(TARGET ${HALIDE_TARGETS})
    halide_add_aot_library_dependency(peak_comparison ${TARGET})
  endforeach()
  target_link_libraries(peak_comparison PRIVATE
    scanner_halide)
endif()
