cmake_minimum_required(VERSION 3.20)
project(caffe)
enable_testing()

# TODO(gc):
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CAFFE_TEST_COPTS -Wall -Wextra -Werror -g)
set(CAFFE_LIB_COPTS -Wall -Wextra -Werror)
set(CAFFE_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/include")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindLibUnwind)
include(FindMKL)
include(FindGlog)
include(FindcuBLAS)

find_package(GTest REQUIRED)

# include_directories(${CMAKE_SOURCE_DIR}/include/)
# include_directories(/usr/local/cuda/include)
# include_directories(/opt/intel/oneapi/mkl/latest/include)
# link_directories(/usr/local/cuda/lib64)
# link_directories(/opt/intel/oneapi/mkl/latest/lib)
# link_directories(/opt/intel/oneapi/compiler/latest/lib)

# find_package(GTest) find_package(glog)

add_subdirectory(src)
add_subdirectory(test)
# add_subdirectory(example)
