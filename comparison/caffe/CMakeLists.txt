set(LIBRARIES
  "${Boost_LIBRARIES}")

if (BUILD_CAFFE_EVALUATORS)
  add_executable(caffe_throughput
    main.cpp
    ${PROTO_SRCS}
    $<TARGET_OBJECTS:evaluators>
    $<TARGET_OBJECTS:video>
    $<TARGET_OBJECTS:util>)

  target_link_libraries(caffe_throughput
    "${LIBRARIES}"
    "${PUBLIC_LIBRARIES}"
    "${SCANNER_LIBRARIES}"
    "${PLATFORM_LINK_FLAGS}")
endif()
