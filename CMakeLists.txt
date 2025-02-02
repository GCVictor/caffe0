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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  message(STATUS "Target architecture: x64")
  set(ARCH_X64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
  message(STATUS "Target architecture: ARM (32-bit)")
  set(ARCH_ARM32 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message(STATUS "Target architecture: ARM (64-bit)")
  set(ARCH_ARM64 TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
  message(STATUS "Target architecture: MIPS")
  set(ARCH_MIPS TRUE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
  message(STATUS "Target architecture: RISC-V")
  set(ARCH_RISCV TRUE)
else()
  message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(example)
