include(CMakeParseArguments)

function(halide_project name folder)
  add_executable("${name}" ${ARGN})
  if (MSVC)
  else()
    target_compile_options("${name}" PUBLIC "-std=c++11") # Halide clients need C++11
    if(NOT HALIDE_ENABLE_RTTI)
      target_compile_options("${name}" PUBLIC "-fno-rtti")
    endif()
  endif()
  target_link_libraries("${name}" PRIVATE "${HALIDE_ROOT_DIR}/lib/libHalide.a" dl pthread z rt tinfo)
  target_include_directories("${name}" PRIVATE "${HALIDE_ROOT_DIR}/include")
  target_include_directories("${name}" PRIVATE "${HALIDE_ROOT_DIR}/tools")
  set_target_properties("${name}" PROPERTIES FOLDER "${folder}")
  if (MSVC)
    # 4006: "already defined, second definition ignored"
    # 4088: "/FORCE used, image may not work"
    # (Note that MSVC apparently considers 4088 too important to allow us to ignore it;
    # I'm nevertheless leaving this here to document that we don't care about it.)
    set_target_properties(${name} PROPERTIES LINK_FLAGS "/ignore:4006 /ignore:4088")
    target_compile_definitions("${name}" PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_link_libraries("${name}" PRIVATE Kernel32)
  endif()
endfunction(halide_project)

function(halide_generator_genfiles_dir NAME OUTVAR)
  set(GENFILES_DIR "${CMAKE_BINARY_DIR}/generator_genfiles/${NAME}")
  file(MAKE_DIRECTORY "${GENFILES_DIR}")
  set(${OUTVAR} "${GENFILES_DIR}" PARENT_SCOPE)
endfunction()

function(halide_generator_get_exec_path TARGET OUTVAR)
  if(MSVC)
    # In MSVC, the generator executable will be placed in a configuration specific
    # directory specified by ${CMAKE_CFG_INTDIR}.
    set(${OUTVAR} "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}" PARENT_SCOPE)
  elseif(XCODE)
    # In Xcode, the generator executable will be placed in a configuration specific
    # directory, so the Xcode variable $(CONFIGURATION) is passed in the custom build script.
    set(${OUTVAR} "${CMAKE_BINARY_DIR}/$(CONFIGURATION)/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}" PARENT_SCOPE)
  else()
    get_target_property(GENERATOR_FOLDER ${args_GENERATOR_TARGET} FOLDER)
    set(${OUTVAR} "${GENERATOR_FOLDER}/${TARGET}${CMAKE_EXECUTABLE_SUFFIX}" PARENT_SCOPE)
  endif()
endfunction()

function(halide_generator_add_exec_generator_target EXEC_TARGET)
  set(options )
  set(oneValueArgs GENERATOR_TARGET GENFILES_DIR)
  set(multiValueArgs OUTPUTS GENERATOR_ARGS)
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  halide_generator_get_exec_path(${args_GENERATOR_TARGET} EXEC_PATH)

  add_custom_command(
    OUTPUT ${args_OUTPUTS}
    DEPENDS ${args_GENERATOR_TARGET}
    COMMAND ${EXEC_PATH} ${args_GENERATOR_ARGS}
    WORKING_DIRECTORY ${args_GENFILES_DIR}
    COMMENT "Executing Generator ${args_GENERATOR_TARGET} with args ${args_GENERATOR_ARGS}..."
  )

  add_custom_target(${EXEC_TARGET} DEPENDS ${args_OUTPUTS})
  set_target_properties(${EXEC_TARGET} PROPERTIES FOLDER "generator")
endfunction()

# This function adds custom build steps to invoke a Halide generator exectuable
# and produce a static library containing the generated code.
#
# The generator executable must be produced separately, e.g. using a call to the
# function halide_add_generator() or halide_project(...) or add_executable(...)
# and passed to this function in the GENERATOR_TARGET parameter.
#
# Usage:
#   halide_add_aot_library(<name>
#                          GENERATOR_TARGET <target>
#                          GENERATOR_NAME <string>
#                          GENERATED_FUNCTION <string>
#                          GENERATOR_OUTPUTS <arg> <arg> ...
#                          GENERATOR_ARGS <arg> <arg> ...)
#
#   <name> is the name of the library being defined.
#   GENERATOR_TARGET is the name of the generator executable target, which is assumed to be
#       defined elsewhere.
#   GENERATOR_TARGET is the name of the generator executable target, which is assumed to be
#       defined elsewhere.
#   GENERATOR_NAME is the registered name of the Halide::Generator derived object
#   GENERATED_FUNCTION is the name of the C function to be generated by Halide, including C++
#       namespace (if any); if omitted, default to GENERATOR_NAME
#   GENERATOR_OUTPUTS are the values to pass to -e; if omitted, defaults to "h static_library"
#   GENERATOR_ARGS are optional extra arguments passed to the generator executable during
#     build.
function(halide_add_aot_library AOT_LIBRARY_TARGET)
  # Parse arguments
  set(options )
  set(oneValueArgs GENERATOR_TARGET GENERATOR_NAME GENERATED_FUNCTION)
  set(multiValueArgs GENERATOR_ARGS GENERATOR_OUTPUTS)
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (args_GENERATED_FUNCTION STREQUAL "")
    set(args_GENERATED_FUNCTION ${args_GENERATOR_NAME})
  endif()

  # Create a directory to contain generator specific intermediate files
  halide_generator_genfiles_dir(${AOT_LIBRARY_TARGET} GENFILES_DIR)

  # Determine the name of the output files
  set(FILTER_LIB "${AOT_LIBRARY_TARGET}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(FILTER_HDR "${AOT_LIBRARY_TARGET}.h")
  set(FILTER_CPP "${AOT_LIBRARY_TARGET}.cpp")

  set(GENERATOR_EXEC_ARGS "-o" "${GENFILES_DIR}")
  if (NOT ${args_GENERATED_FUNCTION} STREQUAL "")
    list(APPEND GENERATOR_EXEC_ARGS "-f" "${args_GENERATED_FUNCTION}" )
  endif()
  if (NOT ${args_GENERATOR_NAME} STREQUAL "")
    list(APPEND GENERATOR_EXEC_ARGS "-g" "${args_GENERATOR_NAME}")
  endif()
  if (NOT "${args_GENERATOR_OUTPUTS}" STREQUAL "")
    string(REPLACE ";" "," _tmp "${args_GENERATOR_OUTPUTS}")
    list(APPEND GENERATOR_EXEC_ARGS "-e" ${_tmp})
  endif()
  # GENERATOR_ARGS always come last
  list(APPEND GENERATOR_EXEC_ARGS ${args_GENERATOR_ARGS})

  if ("${args_GENERATOR_OUTPUTS}" STREQUAL "")
    set(args_GENERATOR_OUTPUTS static_library h)
  endif()

  set(OUTPUTS )

  # This is the CMake idiom for "if foo in list"
  list(FIND args_GENERATOR_OUTPUTS "static_library" _lib_index)
  list(FIND args_GENERATOR_OUTPUTS "h" _h_index)
  list(FIND args_GENERATOR_OUTPUTS "cpp" _cpp_index)

  if (${_lib_index} GREATER -1)
    list(APPEND OUTPUTS "${GENFILES_DIR}/${FILTER_LIB}")
  endif()
  if (${_h_index} GREATER -1)
    list(APPEND OUTPUTS "${GENFILES_DIR}/${FILTER_HDR}")
    set_source_files_properties("${GENFILES_DIR}/${FILTER_HDR}" PROPERTIES GENERATED TRUE)
  endif()
  if (${_cpp_index} GREATER -1)
    list(APPEND OUTPUTS "${GENFILES_DIR}/${FILTER_CPP}")
    set_source_files_properties("${GENFILES_DIR}/${FILTER_HDR}" PROPERTIES GENERATED TRUE)
  endif()

  halide_generator_add_exec_generator_target(
    "${AOT_LIBRARY_TARGET}.exec_generator"
    GENERATOR_TARGET ${args_GENERATOR_TARGET}
    GENERATOR_ARGS   "${GENERATOR_EXEC_ARGS}"
    GENFILES_DIR     ${GENFILES_DIR}
    OUTPUTS          ${OUTPUTS}
  )
endfunction(halide_add_aot_library)

# Usage:
#   halide_add_aot_library_dependency(TARGET AOT_LIBRARY_TARGET)
function(halide_add_aot_library_dependency TARGET AOT_LIBRARY_TARGET)
    halide_generator_genfiles_dir(${AOT_LIBRARY_TARGET} GENFILES_DIR)

    add_dependencies("${TARGET}" "${AOT_LIBRARY_TARGET}.exec_generator")

    set(FILTER_LIB "${AOT_LIBRARY_TARGET}${CMAKE_STATIC_LIBRARY_SUFFIX}")
    target_link_libraries("${TARGET}" PRIVATE "${GENFILES_DIR}/${FILTER_LIB}")
    target_include_directories("${TARGET}" PRIVATE "${GENFILES_DIR}")

    if (WIN32)
      if (MSVC)
        # /FORCE:multiple allows clobbering the halide runtime symbols in the lib
        # linker warnings disabled:
        # 4006: "already defined, second definition ignored"
        # 4088: "/FORCE used, image may not work"
        # (Note that MSVC apparently considers 4088 too important to allow us to ignore it;
        # I'm nevertheless leaving this here to document that we don't care about it.)
        set_target_properties("${TARGET}" PROPERTIES LINK_FLAGS "/STACK:8388608,1048576 /FORCE:multiple /ignore:4006 /ignore:4088")
      else()
        set_target_properties("${TARGET}" PROPERTIES LINK_FLAGS "-Wl,--allow-multiple-definition")
      endif()
    else()
      target_link_libraries("${TARGET}" PRIVATE dl pthread z)
    endif()
endfunction(halide_add_aot_library_dependency)

function(halide_add_generator NAME)
  set(options WITH_STUB)
  set(oneValueArgs STUB_GENERATOR_NAME)
  set(multiValueArgs SRCS STUB_DEPS)
  cmake_parse_arguments(args "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # We need to generate an "object" library for every generator, so that any
  # generator that depends on our stub can link in our generator as well.
  # Unfortunately, an ordinary static library won't do: CMake has no way to
  # force "alwayslink=1", and a static library with just a self-registering
  # Generator is almost certain to get optimized away at link time. Using
  # an "Object Library" lets us dodge this (it basically just groups .o files
  # together and presents them at the end), at the cost of some decidedly
  # ugly bits right here.
  set(OBJLIB "${NAME}.objlib")
  add_library("${OBJLIB}" OBJECT ${args_SRCS})
  # add_dependencies("${OBJLIB}" Halide)
  target_include_directories("${OBJLIB}" PRIVATE "${HALIDE_ROOT_DIR}/include")
  target_compile_options("${OBJLIB}" PRIVATE "-std=c++11" "-fno-rtti")
  foreach(STUB ${args_STUB_DEPS})
    halide_add_generator_stub_dependency(TARGET ${OBJLIB} STUB_GENERATOR_TARGET ${STUB})
  endforeach()

  set(ALLSTUBS $<TARGET_OBJECTS:${OBJLIB}>)
  foreach(STUB ${args_STUB_DEPS})
    list(APPEND ALLSTUBS $<TARGET_OBJECTS:${STUB}.objlib>)
  endforeach()

  halide_project("${NAME}"
                 "generator"
                 "${HALIDE_ROOT_DIR}/tools/GenGen.cpp"
                 ${ALLSTUBS})

  # Declare a stub library if requested.
  if (${args_WITH_STUB})
    halide_add_generator_stub_library(STUB_GENERATOR_TARGET "${NAME}"
                                      STUB_GENERATOR_NAME ${args_STUB_GENERATOR_NAME})
  endif()

  set_target_properties("${NAME}" PROPERTIES FOLDER "${CMAKE_CURRENT_BINARY_DIR}")

  # Add any stub deps passed to us.
endfunction(halide_add_generator)

function(halide_add_generator_stub_library)
  set(options )
  set(oneValueArgs STUB_GENERATOR_TARGET STUB_GENERATOR_NAME)
  set(multiValueArgs )
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  halide_generator_genfiles_dir(${args_STUB_GENERATOR_TARGET} GENFILES_DIR)

  # STUBNAME_BASE = strip_suffix(STUB_GENERATOR_TARGET, ".generator")
  string(REGEX REPLACE "\\.generator*$" "" STUBNAME_BASE ${args_STUB_GENERATOR_TARGET})

  set(STUB_HDR "${GENFILES_DIR}/${STUBNAME_BASE}.stub.h")

  set(GENERATOR_EXEC_ARGS "-o" "${GENFILES_DIR}" "-e" "cpp_stub")
  if (NOT ${args_STUB_GENERATOR_NAME} STREQUAL "")
    list(APPEND GENERATOR_EXEC_ARGS "-g" "${args_STUB_GENERATOR_NAME}")
    list(APPEND GENERATOR_EXEC_ARGS "-n" "${STUBNAME_BASE}")
  endif()

  set(STUBGEN "${args_STUB_GENERATOR_TARGET}.exec_stub_generator")
  halide_generator_add_exec_generator_target(${STUBGEN}
    GENERATOR_TARGET ${args_STUB_GENERATOR_TARGET}
    GENERATOR_ARGS   "${GENERATOR_EXEC_ARGS}"
    GENFILES_DIR     ${GENFILES_DIR}
    OUTPUTS          "${STUB_HDR}"
  )
  set_source_files_properties("${STUB_HDR}" PROPERTIES GENERATED TRUE)
endfunction(halide_add_generator_stub_library)

function(halide_add_generator_stub_dependency)
  # Parse arguments
  set(options )
  set(oneValueArgs TARGET STUB_GENERATOR_TARGET)
  set(multiValueArgs )
  cmake_parse_arguments(args "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  halide_generator_genfiles_dir(${args_STUB_GENERATOR_TARGET} GENFILES_DIR)
  set(STUBGEN "${args_STUB_GENERATOR_TARGET}.exec_stub_generator")
  add_dependencies("${args_TARGET}" ${STUBGEN})
  target_include_directories("${args_TARGET}" PRIVATE "${GENFILES_DIR}")
endfunction(halide_add_generator_stub_dependency)
