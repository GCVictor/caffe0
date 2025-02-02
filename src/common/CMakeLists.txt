set(SOURCES sync_mem.cc blob.cc)
add_library(common ${SOURCES})
target_compile_options(common PRIVATE ${CAFFE_LIB_COPTS})
target_include_directories(common PRIVATE ${CAFFE_SOURCE_INCLUDE_DIRS})
target_include_directories(common PRIVATE ${GLOG_INCLUDE_DIRS})
target_include_directories(common PUBLIC ${PROTO_GEN_DIR})
target_link_libraries(common PRIVATE ${GLOG_LIBRARIES})
target_link_libraries(common PRIVATE protolib)
target_link_libraries(common PRIVATE ${Protobuf_LIBRARIES})

if(USE_CPU)
  target_compile_definitions(common PUBLIC CPU_ONLY)
elseif(CUDA_LIBRARIES)
  target_compile_definitions(common PUBLIC USE_CUDA)
  target_link_libraries(common PRIVATE ${CUDA_LIBRARIES})
elseif(HIP_LIBRARIES)
  target_compile_definitions(common PUBLIC USE_HIP)
endif()

add_library(caffe::common ALIAS common)
