function(add_binary)
    # cmake_parse_arguments(<prefix> <options> <one_value_keywords> <multi_value_keywords> <args>...)
    cmake_parse_arguments(BINARY "" "NAME" "HDRS;SRCS;COPTS;DEPS" ${ARGN})

    add_executable(${BINARY_NAME} ${BINARY_SRCS})

    if(BINARY_HDRS)
        target_include_directories(${BINARY_NAME} PRIVATE ${BINARY_HDRS})
    endif()

    if(BINARY_COPTS)
        target_compile_options(${BINARY_NAME} PRIVATE ${BINARY_COPTS})
    endif()

    if(BINARY_DEPS)
        target_link_libraries(${BINARY_NAME} PRIVATE ${BINARY_DEPS})
    endif()
endfunction()
