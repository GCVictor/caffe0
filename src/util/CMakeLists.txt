caffe_add_library(
NAME
    util
HDRS
    "${MKL_INCLUDE_DIRS}"
    "${CMAKE_SOURCE_DIR}/include" 
SRCS
    "math_functions.cc"
COPTS
    ${CAFFE_LIB_COPTS}
DEPS
    "${MKL_LIBRARIES}"
    "${glog_LIBRARIES}"
ALIAS
    caffe::util
)