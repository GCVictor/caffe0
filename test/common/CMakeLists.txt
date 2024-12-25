set(SOURCES synced_mem_test.cc)

foreach(source ${SOURCES})
  get_filename_component(target_name ${source} NAME_WE)
  add_executable(${target_name} ${source})
  target_include_directories(${target_name} PRIVATE ${CAFFE_INCLUDE_DIRS})
  target_include_directories(${target_name} PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(${target_name} PRIVATE caffe::common)

  target_link_libraries(${target_name} PRIVATE ${GLOG_LIBRARIES})
  target_link_libraries(${target_name} PRIVATE ${GTEST_BOTH_LIBRARIES})
  target_link_libraries(${target_name} PRIVATE ${CUDA_LIBRARIES})
endforeach()
