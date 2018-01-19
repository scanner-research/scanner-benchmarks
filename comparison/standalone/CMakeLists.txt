add_executable(standalone_comparison
  main.cpp
  $<TARGET_OBJECTS:util>)

target_link_libraries(standalone_comparison
  ${LIBRARIES}
  ${CMAKE_CURRENT_SOURCE_DIR}/generator_genfiles/caffe_input_transformer_cpu/caffe_input_transformer_cpu.a
  ${CMAKE_CURRENT_SOURCE_DIR}/generator_genfiles/caffe_input_transformer_gpu/caffe_input_transformer_gpu.a)
