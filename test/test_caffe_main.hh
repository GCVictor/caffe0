#pragma once

// #include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

namespace caffe {

// cudaDeviceProp CAFFE_TEST_CUDA_PROP;

typedef ::testing::Types<float, double> TestDtypes;

}  // namespace caffe
