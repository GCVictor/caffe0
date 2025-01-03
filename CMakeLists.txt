cmake_minimum_required(VERSION 3.20)
project(caffe)
enable_testing()

# TODO(gc):
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CAFFE_TEST_COPTS -Wall -Wextra -Werror -g)
set(CAFFE_LIB_COPTS -Wall -Wextra -Werror)
set(CAFFE_SOURCE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")
set(CAFFE_TEST_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/test")
set(PROTO_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${PROTO_GEN_DIR})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindLibUnwind)
include(FindMKL)
include(FindGlog)
include(FindcuBLAS)

find_package(GTest REQUIRED)
find_package(Protobuf REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(example)
